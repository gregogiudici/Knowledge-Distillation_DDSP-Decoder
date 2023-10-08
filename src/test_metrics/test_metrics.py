import numpy as np
import torch.fft as fft
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
from torch import nn
import statistics

'''
Log Spectral Distance
'''
def log_spectral_distance(ref, synth, p=1, mode='mean'):
    """    
    Compute the Log-Spectral Distance (LSD) between two signals.
    https://www.wikiwand.com/en/Log-spectral_distance
    
    Parameters
    ----------
    ref : array_like
        reference signal.
    synth : array_like
        synthesized signal.
    p : int
        pnorm used to compute the output:
            p = 1 -> L1\n
            p = 2 -> L2
    mode : str
        if ref and synth have size [B,N], mode defines the output shape.\n
        if mode = 'sum' returns the sum;\n 
        if mode = 'mean' returns the mean;\n
        if mode = null returns an array-like vector with the loss  calculated for each batch.\n
        if ref and synth have size [1,N] all three mode returns the same value.\n
    
    Returns
    --------
    loss : lsd between ref and synth
    """
    
    if(len(ref.size()) == 3):
        ref = ref.squeeze(-1)
    if(len(synth.size()) == 3):
        synth = synth.squeeze(-1)
    
    ref = ref.detach().cpu().numpy()
    synth = synth.detach().cpu().numpy()
    
    # Compute FFT
    fft_ref = np.fft.fft(ref)
    fft_synth = np.fft.fft(synth)

    # Compute spectral magnitude
    mag_ref = np.abs(fft_ref)
    mag_synth = np.abs(fft_synth)
    
    loss = []
    for s_ref, s_synth in zip(mag_ref, mag_synth):
        if(p == 2): # L2 distance
            loss.append(np.sqrt(np.mean((np.log(s_ref) - np.log(s_synth) ** 2))))
        else: # L1 distance 
            loss.append((np.mean(np.abs(np.log(s_ref) - np.log(s_synth)))))
            
    # define output
    if(mode == 'mean'):  
        loss = sum(loss) / len(loss)
    if(mode == 'sum'):
        loss = sum(loss)
        
    return loss

'''
Spectral Onset Flux distance
'''
def spec_onset_flux_distance(ref, synth, sample_rate=16000, mode='mean'):
    """    
    Compute the difference between the Spectral Onset Flux of two signals.
    https://www.eecs.qmul.ac.uk/~simond/pub/2006/dafx.pdf
    
    Parameters
    ----------
    ref : array_like
        reference signal.
    synth : array_like
        synthesized signal.
    sample_rate : int
        sample rate of the two signals
    mode : str
        if ref and synth have size [B,N], mode defines the output shape.\n
        if mode = 'sum' returns the sum;\n 
        if mode = 'mean' returns the mean;\n
        if mode = null returns an array-like vector with the loss calculated for each batch.\n
        if ref and synth have size [1,N] all three mode returns the same value.\n
    
    Returns
    -------
    loss : float
        difference between onset_ref and onset_synth
    onset_ref : array-like
        spetral onset flux of the reference signal
    onset_synth : array-like
        spectral onset flux of the synth signal
    ---------
    
    """
    if(len(ref.size()) == 3):
        ref = ref.squeeze(-1)
    if(len(synth.size()) == 3):
        synth = synth.squeeze(-1)
        
    ref = ref.detach().cpu().numpy()
    synth = synth.detach().cpu().numpy()
    
    onset_ref = []
    onset_synth = []
    loss = []
    for r, s in zip(ref, synth):
        flux_r = (librosa.onset.onset_strength(y=r, sr=sample_rate))
        flux_s = (librosa.onset.onset_strength(y=s, sr=sample_rate))
        loss.append(np.mean(np.abs(flux_r - flux_s)))
        onset_ref.append(flux_r)
        onset_synth.append(flux_s)
    
    # define output
    if(mode == 'mean'):  
        loss = sum(loss) / len(loss)
    if(mode == 'sum'):
        loss = sum(loss)
            
    return loss, onset_ref, onset_synth

        
