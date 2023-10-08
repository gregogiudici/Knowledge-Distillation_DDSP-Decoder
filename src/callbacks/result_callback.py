from typing import Any, Optional
import lightning.pytorch as pl
from lightning.pytorch import LightningModule
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import Callback
import h5py
import os.path as path
from lightning.pytorch.utilities.types import STEP_OUTPUT
import soundfile as sf
import numpy as np
import os

class ResultSaver(Callback):
    """Lightning Callaback to save model results after test.

    Parameters
    ----------
    instrument: str
        name of the instrument used to test the model
    sample_rate: int
        sample rate used by synth and by audio data
    frame_rate: int
        frame rate used to train the decoder
    dirpath: str
        directory where store the files
    val_name: str [optional]
        if want to save also validation results
    test_name: str
        part of file name used to test the model
    
    """
    def __init__(self,
                instrument : str,
                sample_rate : int,
                frame_rate : int,
                dirpath : str,
                val_name="val",
                test_name="test"):
        self.dirpath = dirpath+"results_val_test"
        self.instrument = instrument
        self.val_file = val_name
        self.test_file = test_name
        # self.step_count = 0
        self.val_store_dict = {}
        self.test_store_dict = {}
        self.teacher_store_dict = {}
        self.student_store_dict = {}
        self.sr = sample_rate
        self.fr = frame_rate
        self.distillation = False
        # Create directory to store results
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)
               
    def store_step(self, data, store_dict):
        if not bool(store_dict):
            for x in data:
                for k in x.keys():
                    if(k == 'audio' or k == 'synth_audio'):
                        data = x[k].squeeze(-1).reshape(-1)
                    else:
                        data = x[k]
                    store_dict[k] = data.detach().cpu().numpy()
        else:
            for x in data:
                for k in x.keys():
                    if(k == 'audio' or k == 'synth_audio'):
                        data = x[k].squeeze(-1).reshape(-1)
                        concat_dim = 0
                    else:
                        data = x[k]
                        concat_dim = -3
                    store_dict[k] = np.concatenate(
                                    (store_dict[k],data.detach().cpu().numpy()),
                                    axis=concat_dim)
        return store_dict

    def save_and_close(self, store_dict, data_dir, filename):
        #print("[DEBUG] ResultCallback: Saving {}_{}_{}sr_{}fr.h5".format(filename, self.instrument, self.sr, self.fr))
        h5f = h5py.File(f'{data_dir}/{filename}_{self.instrument}_{self.sr}sr_{self.fr}fr.h5', 'w')
        #print("\t self.store_dict.keys() {}".format(self.store_dict.keys()))
        #print("\t self.store_dict['audio'] {}".format(self.store_dict['audio'].shape))
        #print("\t self.store_dict['synth_audio'] {}".format(self.store_dict['synth_audio'].shape))
        for k in store_dict.keys():
            h5f.create_dataset(k, data=store_dict[k])
        h5f.close()
        
        sf.write(
            path.join(self.dirpath, f"{self.instrument}_ref_{filename}.wav"),
            np.squeeze(store_dict['audio']),
            self.sr,
        )
        sf.write(
            path.join(self.dirpath, f'{self.instrument}_synth_{filename}.wav'),
            np.squeeze(store_dict['synth_audio']),
            self.sr,
        )
      
    # VALIDATION    
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int):  
        # x = batch
        # x_hat = outputs
        # self.val_store_dict = self.store_step([x, x_hat], self.val_store_dict)
        # print("[DEBUG] on_validation_batch_end {}".format(batch_idx))
        pass
        
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        #self.save_and_close(self.val_store_dict, self.dirpath, self.val_file)  
        #print("[DEBUG] on_validation_END") 
        pass

    # TEST    
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int):
        x = batch
        if type(outputs) is tuple:
            self.distillation = True
            t_hat, s_hat = outputs
            self.test_store_dict = self.store_step([x, t_hat], self.teacher_store_dict)
            self.test_store_dict = self.store_step([x, s_hat], self.student_store_dict)
        else:
            x_hat = outputs
            self.test_store_dict = self.store_step([x, x_hat], self.test_store_dict)
        # print("[DEBUG] on_test_batch_end  {}".format(batch_idx)) 
        
    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.distillation:
            self.save_and_close(self.teacher_store_dict, self.dirpath, "teacher")
            self.save_and_close(self.student_store_dict, self.dirpath, "student")  
        else:
            self.save_and_close(self.test_store_dict, self.dirpath, self.test_file)  
        # print("[DEBUG] on_test_END") 
