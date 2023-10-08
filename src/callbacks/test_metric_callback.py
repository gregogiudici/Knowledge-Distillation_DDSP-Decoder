from typing import Any, Optional
import lightning.pytorch as pl
from lightning.pytorch import LightningModule
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

import librosa
import numpy as np
from src.test_metrics import log_spectral_distance, spec_onset_flux_distance

class TestMetricsCallback(Callback):
    """Lightning Callaback to calculate metrics during test.\n
    the metrics used are:
    - Log Spectral Distance
    - Spectral Onset Flux distance\n
    Parameters
    ----------
    mode_lsd: str ['mean', 'sum', null]\n
        if ref and synth have size [B,N], mode_lsd defines the output shape from the Log Spectral Distance function.\n
        if mode = 'sum' returns the sum;\n 
        if mode = 'mean' returns the mean;\n
        if mode = null returns an array-like vector with the loss  calculated for each batch.\n
        if ref and synth have size [1,N] all three mode returns the same value.
    mode_flux: str ['mean', 'sum', null]\n
        if ref and synth have size [B,N], mode_flux defines the output shape from the Spectral Onset Flux Distance function.\n
        if mode = 'sum' returns the sum;\n 
        if mode = 'mean' returns the mean;\n
        if mode = null returns an array-like vector with the loss  calculated for each batch.\n
        if ref and synth have size [1,N] all three mode returns the same value.

    """
    def __init__(self, mode_lsd='mean', mode_flux='mean'):
        self.mode_lsd = mode_lsd
        self.mode_flux = mode_flux
        
    # VALIDATION    
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int):  
        pass
        
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule): 
        pass

    # TEST    
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int):
        x = batch
        if type(outputs) is tuple:
            t_hat, s_hat = outputs
            sr = pl_module.get_sr()
            s_lsd = log_spectral_distance(x['audio'], s_hat['synth_audio'], p=1, mode= self.mode_lsd)
            t_lsd = log_spectral_distance(x['audio'], t_hat['synth_audio'], p=1, mode= self.mode_lsd)
            t_flux_diff, onset_x, onset_t_hat = spec_onset_flux_distance(x['audio'], t_hat['synth_audio'], sr, mode=self.mode_flux)
            s_flux_diff, onset_x, onset_s_hat = spec_onset_flux_distance(x['audio'], s_hat['synth_audio'], sr, mode=self.mode_flux)
            # Log results
            self.log("test/teacher_lsd", t_lsd, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/teacher_flux_diff", t_flux_diff, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/student_lsd", s_lsd, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/student_flux_diff", s_flux_diff, on_step=False, on_epoch=True, prog_bar=True)
        else:
            x_hat = outputs
            sr = pl_module.get_sr()
            lsd = log_spectral_distance(x['audio'], x_hat['synth_audio'], p=1, mode= self.mode_lsd)
            flux_diff, onset_x, onset_x_hat = spec_onset_flux_distance(x['audio'], x_hat['synth_audio'], sr, mode=self.mode_flux)
            # Log results
            self.log("test/lsd", lsd, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/flux_diff", flux_diff, on_step=False, on_epoch=True, prog_bar=True)
            #print(np.shape(flux_loss), np.shape(onset_x), np.shape(onset_x_hat))
            
    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pass
