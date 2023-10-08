from typing import Any, Optional
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion

import torch
import torch.nn as nn
from lightning import LightningModule
from src.models import S4decoder

class DDSPDecoderTrainTask(LightningModule):
    """LightningModule for DDSP Decoder.

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
        decoder: nn.Module,
        synth: nn.Module,
        preprocessor,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        scheduler_steps: int
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # ATTENTION: 
        # we must set ignore=["decoder"] to prevent:
        #       RuntimeError: Cannot save multiple tensors or storages that view the same data as different types.
        # with S4decoder with mode='nplr'
        self.save_hyperparameters(ignore=["decoder","synth","criterion"],logger=False)          
        
        
        # defining the net
        net = []
        net.append(decoder)
        net.append(synth)
        self.net = nn.Sequential(*net)
        
        # defining data preprocessor
        self.preprocessor = preprocessor
                
        # loss function
        self.criterion = criterion
        
        # scheduler frequency step
        self.scheduler_steps = scheduler_steps
        
        # training loss, validation loss, test loss
        self.train_loss = 0
        self.val_loss = 0
        self.test_loss = 0
        
    def get_sr(self):
        '''Get sample rate used by synthesizer'''
        return self.net[1].sample_rate

    def forward(self, x: torch.Tensor):
        x = self.preprocessor.run(x)
        x_hat = self.net(x)
        return x_hat
    
    def model_step(self, batch: Any):
        x = batch
        x_hat = self.forward(batch)
        loss = self.criterion(x['audio'], x_hat['synth_audio'])
        return loss, x_hat
    
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss = 0

    def training_step(self, batch: Any, batch_idx: int):
        loss, x_hat = self.model_step(batch)

        # update and log metrics
        self.train_loss = loss
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        #sch = self.lr_schedulers()
        #print("Learning Rate",sch.get_lr())
        
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, x_hat = self.model_step(batch)

        # update and log metrics
        self.val_loss = loss
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return x_hat

    def on_validation_epoch_end(self):
        #acc = self.val_acc.compute()  # get current val acc
        #self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        #self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, x_hat = self.model_step(batch)

        # update and log metrics
        self.test_loss = loss
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        return x_hat

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
        if isinstance(self.net[0],S4decoder):
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
            return super().lr_scheduler_step(scheduler, metric)


if __name__ == "__main__":
    _ = DDSPDecoderTrainTask(None, None, None)
