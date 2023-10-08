import torch
import torch.nn as nn
from functools import partial

def safe_log(x,eps=1e-7):
    eps = torch.tensor(eps)
    return torch.log(x + eps)

def multiscale_fft(signal, scales, overlap):
    """    
    Compute Multi-Scale Short Time Fourier Transform
    
    Parameters
    ----------
    signal : array_like
        time signal.
    scales : [int]
        list of scales used to compute different stfts
    overlap : float
        overlap
    Returns
    -------
    stfts : [stft]
        list of signal's stfts computed for each scale
    ---------
    
    """
    stfts = []
    for s in scales:
        S = torch.stft(
            signal,
            s,
            int(s * (1 - overlap)),
            s,
            torch.hann_window(s).to(signal),
            True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts


def asim_l1_distance(a,b,alpha=1,beta=1):
    ''' Asimetric L1 distance '''
    diff = a-b
    pos_diff = diff * (diff > 0)
    neg_diff = diff * (diff < 0)
    as_diff = alpha * pos_diff + beta * neg_diff
    as_mse = torch.abs(as_diff).mean()
    return as_mse


def asim_msfft_loss(a1,
                    a2,
                    scales=[4096, 2048, 1024, 512, 256, 128],
                    overlap=0.75,
                    alpha=1,
                    beta=1):
    '''
    DDSP Original MS FFT loss with lin + log spectra analysis
    '''
    if(len(a1.size()) == 3):
        a1 = a1.squeeze(-1)
    if(len(a2.size()) == 3):
        a2 = a2.squeeze(-1)
    ori_stft = multiscale_fft(
        a1,
        scales,
        overlap,
    )
    rec_stft = multiscale_fft(
        a2,
        scales,
        overlap,
    )

    loss = 0
    for s_x, s_y in zip(ori_stft, rec_stft):
        lin_loss = asim_l1_distance(s_x, s_y,alpha,beta)
        log_loss = asim_l1_distance(safe_log(s_x),safe_log(s_y),alpha,beta)
        loss = loss + lin_loss + log_loss

    return loss



def ddsp_msfft_loss(a1,
                    a2,
                    scales=[4096, 2048, 1024, 512, 256, 128],
                    overlap=0.75):
    '''
    DDSP Original MS FFT loss with lin + log spectra analysis
        Some remarks: the stfts have to be normalized otherwise the network weights different excerpts to different importance.
                      We compute the mean of the L1 difference between normalized magnitude spectrograms
                      so that the magnitude of the loss do not change with the window size.
    '''
    if(len(a1.size()) == 3):
        a1 = a1.squeeze(-1)
    if(len(a2.size()) == 3):
        a2 = a2.squeeze(-1)
    ori_stft = multiscale_fft(
        a1,
        scales,
        overlap,
    )
    rec_stft = multiscale_fft(
        a2,
        scales,
        overlap,
    )

    loss = 0
    for s_x, s_y in zip(ori_stft, rec_stft):
        lin_loss = (s_x - s_y).abs().mean()
        log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
        loss = loss + lin_loss + log_loss

    return loss


class rec_loss(nn.Module):
    '''
    Module used to compute the Reconstruction Loss in an Autoencoder:
    MS FFT loss with lin + log spectral distance
        
    Parameters
    ----------
    scales : [int]
        list of scales used to compute different stfts
    overlap : float
        overlap
    alpha / beta : float
        parameters for asimetrical L1 distance
    
    Forward:
    -------
    INPUT :
    ref : reference signal 
    synth: recostructed signal\n
    OUTPUT :
    loss : reconstruction loss
    ---------
    '''
    def __init__(self, scales, overlap, alpha=None, beta=None):
        super().__init__()
        self.scales = scales
        self.overlap = overlap
        if(alpha is not None and beta is not None):
            self.loss_fn = partial(asim_msfft_loss, alpha=alpha, beta=beta)
            print(f'[INFO] rec_loss() - Using asimetrical reconstruction loss. alpha: {alpha} - beta: {beta}')
        else:
            self.loss_fn = ddsp_msfft_loss
    def forward(self,ref,synth):
        return self.loss_fn(ref, 
                            synth,
                            self.scales,
                            self.overlap)

class distillation_loss(nn.Module):
    '''
    Module used to compute and combine reconstruction loss with Knowledge Distillation:
    MS FFT loss with lin + log spectral distance
        
    Parameters
    ----------
    scales : [int]
        list of scales used to compute different stfts
    overlap : float
        overlap
    alpha / beta : float
        parameters for asimetrical L1 distance
    
    Forward:
    -------
    INPUT :
    ref : reference signal 
    student_synth: recostructed signal by student\n
    teacher_synth: recostructed signal by teacher\n
    OUTPUT :
    loss : reconstruction loss (student-reference)
    distillation_loss : recostruntion loss (student-teacher)
    loss1 : weighted recostruction loss (student-reference)
    loss2 : weighted recostruction loss (student-teacher)
    ---------
    '''
    def __init__(self, scales, overlap, alpha=None, beta=None):
        super().__init__()
        self.log_vars = nn.parameter.Parameter(torch.zeros(2))
        self.scales = scales
        self.overlap = overlap
        if(alpha is not None and beta is not None):
            self.loss_fn = partial(asim_msfft_loss, alpha=alpha, beta=beta)
            print(f'[INFO] rec_loss() - Using asimetrical reconstruction loss. alpha: {alpha} - beta: {beta}')
        else:
            self.loss_fn = ddsp_msfft_loss
            
    def forward(self, ref, teacher_synth, student_synth):
        
        loss = self.loss_fn(ref, student_synth, self.scales, self.overlap)
        distillation_loss = self.loss_fn(teacher_synth, student_synth, self.scales, self.overlap)

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss + self.log_vars[0]
        
        precision1 = torch.exp(-self.log_vars[0])
        loss1 = precision1*distillation_loss + self.log_vars[0]
        
        return loss, distillation_loss, loss0, loss1
