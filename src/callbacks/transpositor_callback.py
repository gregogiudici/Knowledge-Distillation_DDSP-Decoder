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

class TranspositorCallback(Callback):
    """Lightning Callaback to compute Tonality Transfer.

    Parameters
    ----------
    transpose_factor : int
        transposition multiplier
    instrument: str
        name of the instrument used to test the model
    sample_rate: int
        sample rate used by synth and by audio data
    frame_rate: int
        frame rate used to train the decoder
    dirpath: str
        directory where store the files
    """
    def __init__(self,
                instrument : str,
                sample_rate : int,
                frame_rate : int,
                dirpath : str,
                name="transpose",
                transpose_factor : int = 2):
        
        self.dirpath = dirpath+"results_val_test"
        self.instrument = instrument
        self.file = name

        self.test_store_dict = {}
        self.sr = sample_rate
        self.fr = frame_rate
        self.transp = transpose_factor
        
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
        sf.write(
            path.join(self.dirpath, f'{self.instrument}_synth_{filename}.wav'),
            np.squeeze(store_dict['synth_audio']),
            self.sr,
        )
      
    # TEST    
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int):
        x = batch
        f0 = x['f0']
        x['f0'] = self.transp*f0
        x_hat = pl_module.forward(x)
        self.test_store_dict = self.store_step([x, x_hat], self.test_store_dict)
        
    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.save_and_close(self.test_store_dict, self.dirpath, self.file)  
