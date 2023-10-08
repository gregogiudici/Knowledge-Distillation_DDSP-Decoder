from typing import Any, Optional
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion

import torch
import torch.nn as nn
from lightning import LightningModule
from src.models import S4decoder
import os
import copy

def rename_keys(mydict):
        return dict((k.removeprefix('net.'), rename_keys(v) if hasattr(v,'keys') else v) for k,v in mydict.items())
    
class KnowledgeDistillationTask(LightningModule):
    """LightningModule for Knowledge Distillation.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        path,
        student_decoder: nn.Module,
        student_synth: nn.Module,
        teacher_decoder: nn.Module,
        teacher_synth: nn.Module,
        preprocessor,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        scheduler_steps: int,
        distill_weights_update: int = 1
    ):
        super().__init__()  
        
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # ATTENTION: 
        # we must set ignore=["student_decoder"] to prevent:
        #       RuntimeError: Cannot save multiple tensors or storages that view the same data as different types.
        # with S4decoder with mode='nplr'
        self.save_hyperparameters(ignore=["student_decoder","student_synth","teacher_decoder","teacher_synth","criterion"],logger=False) 
        
        # defining the student
        student = []
        student.append(student_decoder)
        student.append(student_synth)
        self.student = nn.Sequential(*student)
        
        # defining the teacher
        teacher = []
        teacher.append(teacher_decoder)
        teacher.append(teacher_synth)
        self.teacher = nn.Sequential(*teacher)

        # load teacher
        print(f">>>>>>> LOADING TEACHER MODEL: {path} <<<<<<<<<<\n")
        state_dict = torch.load(os.path.join(path, "checkpoints/best.ckpt"))       
        self.teacher.load_state_dict(rename_keys(state_dict['state_dict']))
        for param in self.teacher.parameters():
            param.requires_grad = False
        
         # defining data preprocessor
        self.preprocessor = preprocessor
        
        # loss function
        self.criterion = criterion
        
        # scheduler frequency step
        self.scheduler_steps = scheduler_steps

        # total loss weight (Knowledge Distillation)
        self.alfa = 0.5
        self.beta = 0.5
        # weights update step
        self.weights_update = distill_weights_update
        
        # training loss, validation loss, test loss
        self.train_loss = 0
        self.val_loss = 0
        self.test_loss = 0

    def get_sr(self):
        '''Get sample rate used by synthesizer'''
        return self.teacher[1].sample_rate

    def forward(self, x: torch.Tensor):
        x = self.preprocessor.run(x)
        self.teacher.eval()
        with torch.no_grad():
            t_hat = self.teacher(x)      
        self.student.train()
        s_hat = self.student(x)
        return t_hat, s_hat
    
    def model_step(self, batch: Any):
        x = batch
        t_hat, s_hat = self.forward(batch)
        loss = self.criterion(x['audio'], s_hat['synth_audio'])
        distillation_loss = self.criterion(t_hat['synth_audio'], s_hat['synth_audio'])
        total_loss = self.beta*loss + self.alfa*distillation_loss
        teacher_loss = self.criterion(x['audio'], t_hat['synth_audio'])

        return total_loss, distillation_loss, t_hat, loss, s_hat, teacher_loss
    
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss = 0
        

    def training_step(self, batch: Any, batch_idx: int):
        total_loss, distillation_loss, t_hat, loss, s_hat, teacher_loss = self.model_step(batch)

        # update and log metrics
        self.train_loss = loss
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/distillation_loss", distillation_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
                
        # return loss or backpropagation will fail
        return total_loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        total_loss, distillation_loss, t_hat, loss, s_hat, teacher_loss = self.model_step(batch)

        # update and log metrics
        self.val_loss = loss
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/distillation_loss", distillation_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        return t_hat, s_hat

    def on_validation_epoch_end(self):
        #acc = self.val_acc.compute()  # get current val acc
        #self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        #self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)
        pass

    def test_step(self, batch: Any, batch_idx: int):
        total_loss, distillation_loss, t_hat, loss, s_hat, teacher_loss = self.model_step(batch)

        # update and log metrics
        self.test_loss = loss
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/teacher_loss", teacher_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/distillation_loss", distillation_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        return t_hat, s_hat

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
      
        ATTENTION:
        S4 requires a specific optimizer setup.

        The S4 layer (A, B, C, dt) parameters typically
        require a smaller learning rate (typically 0.001), with no weight decay.

        The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
        and weight decay (if desired).
        """
        if isinstance(self.student[0],S4decoder):
            all_parameters = list(self.parameters())
            # General parameters don't contain the special _optim key
            params = [p for p in all_parameters if not hasattr(p, "_optim")]
            
            # Create an optimizer with the general parameters
            optimizer = self.hparams.optimizer(params)  
            
            # Add parameters with special hyperparameters
            hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
            hps = [
                    dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
                    ]  # Unique dicts
            for hp in hps:
                params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
                optimizer.add_param_group(
                {"params": params, **hp}
                )                     
        else:
            optimizer = self.hparams.optimizer(params=self.parameters())
           
        # Create a lr scheduler 
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val/loss",
                        "interval": "step",
                        "frequency": 1,
                        "strict": True,
                    },
                }
            
        return {"optimizer": optimizer}
    
  
    def lr_scheduler_step(self, scheduler, metric):
        '''There is a bug while using scheduler.interval = 'step'.
        If we set scheduler frequency > dataset.epoch_step (for example: 5 for flute dataset) 
        inside scheduler configuration, it doesn't work. 
        We must override lr_scheduler_step()'''
        if self.global_step % self.scheduler_steps == 0:
            self.alfa = self.weights_update*self.alfa
            self.beta = 1-self.alfa
            print(f"Distillation weights: alfa={self.alfa}  beta={self.beta}")
            return super().lr_scheduler_step(scheduler, metric)
    
  
if __name__ == "__main__":
    _ = KnowledgeDistillationTask(None, None, None)
