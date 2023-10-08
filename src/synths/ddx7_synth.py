
import torch
import torch.nn as nn
import numpy as np
from src.synths.hpn_synth import *


class FMSynth(nn.Module):
    def __init__(self,
                 sample_rate,
                 block_size,
                 fr=[1,1,1,1,3,14],
                 max_ol=2,
                 scale_fn = torch.sigmoid,
                 synth_module='fmstrings'):
        super().__init__()
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.reverb = Reverb(length=sample_rate, sample_rate=sample_rate)
        fr = torch.tensor(fr) # Frequency Ratio
        self.register_buffer("fr", fr) #Non learnable but sent to GPU if declared as buffers, and stored in model dictionary
        self.scale_fn = scale_fn
        self.use_cumsum_nd = False
        self.max_ol = max_ol

        available_synths = {
            'fmbrass': fm_brass_synth,
            'fmflute': fm_flute_synth,
            'fmstrings': fm_string_synth,
            'fmablbrass': fm_ablbrass_synth,
            '2stack2': fm_2stack2,
            '1stack2':fm_1stack2,
            '1stack4': fm_1stack4}

        self.synth_module = available_synths[synth_module]

    def forward(self,controls):

        ol = self.max_ol*self.scale_fn(controls['ol'])
        ol_up = upsample(ol, self.block_size,'linear')
        f0_up = upsample(controls['f0_hz'], self.block_size,'linear')
        signal = self.synth_module(f0_up,
                                ol_up,
                                self.fr,
                                self.sample_rate,
                                self.max_ol,
                                self.use_cumsum_nd)
        # Reverb part
        synth_signal = self.reverb(signal)

        synth_out = {
            'synth_audio': synth_signal,
            'dereverb_audio' : signal,
            'ol': ol,
            'f0_hz': controls['f0_hz']
            }
        return synth_out

OP6=5
OP5=4
OP4=3
OP3=2
OP2=1
OP1=0

def fm_2stack2(pitch, ol, fr, sample_rate,max_ol,use_safe_cumsum=False):

    if(use_safe_cumsum==True):
        omega = cumsum_nd(2 * np.pi * pitch / sample_rate, 2*np.pi)
    else:
        omega = torch.cumsum(2 * np.pi * pitch / sample_rate, 1)

    # Torch unsqueeze with dim -1 adds a new dimension at the end of ol to match phases.

    op4_phase =  fr[OP4] * omega
    op4_output = torch.unsqueeze(ol[:,:,OP4], dim=-1) * torch.sin(op4_phase)

    op3_phase =  fr[OP3] * omega + 2 * np.pi * op4_output
    op3_output = torch.unsqueeze(ol[:,:,OP3], dim=-1) * torch.sin(op3_phase) # output of stack of 2

    op2_phase =  fr[OP2] * omega
    op2_output = torch.unsqueeze(ol[:,:,OP2], dim=-1) * torch.sin(op2_phase)

    op1_phase =  fr[OP1] * omega + 2 * np.pi * op2_output
    op1_output = torch.unsqueeze(ol[:,:,OP1], dim=-1) * torch.sin(op1_phase) # output of stack of 2

    return (op3_output + op1_output)/max_ol

def fm_1stack2(pitch, ol, fr, sample_rate,max_ol,use_safe_cumsum=False):

    if(use_safe_cumsum==True):
        omega = cumsum_nd(2 * np.pi * pitch / sample_rate, 2*np.pi)
    else:
        omega = torch.cumsum(2 * np.pi * pitch / sample_rate, 1)

    # Torch unsqueeze with dim -1 adds a new dimension at the end of ol to match phases.

    op2_phase =  fr[OP2] * omega
    op2_output = torch.unsqueeze(ol[:,:,OP2], dim=-1) * torch.sin(op2_phase)

    op1_phase =  fr[OP1] * omega + 2 * np.pi * op2_output
    op1_output = torch.unsqueeze(ol[:,:,OP1], dim=-1) * torch.sin(op1_phase) # output of stack of 2

    return op1_output/max_ol


def fm_1stack4(pitch, ol, fr, sample_rate,max_ol,use_safe_cumsum=False):

    if(use_safe_cumsum==True):
        omega = cumsum_nd(2 * np.pi * pitch / sample_rate, 2*np.pi)
    else:
        omega = torch.cumsum(2 * np.pi * pitch / sample_rate, 1)

    # Torch unsqueeze with dim -1 adds a new dimension at the end of ol to match phases.

    op4_phase =  fr[OP4] * omega
    op4_output = torch.unsqueeze(ol[:,:,OP4], dim=-1) * torch.sin(op4_phase)

    op3_phase =  fr[OP3] * omega + 2 * np.pi * op4_output
    op3_output = torch.unsqueeze(ol[:,:,OP3], dim=-1) * torch.sin(op3_phase) # output of stack of 4

    op2_phase =  fr[OP2] * omega + 2 * np.pi * op3_output
    op2_output = torch.unsqueeze(ol[:,:,OP2], dim=-1) * torch.sin(op2_phase)

    op1_phase =  fr[OP1] * omega + 2 * np.pi * op2_output
    op1_output = torch.unsqueeze(ol[:,:,OP1], dim=-1) * torch.sin(op1_phase) # output of stack of 2

    return op1_output/max_ol


'''
Ablated Brass FM Synth - with phase wrapping (it does not change behaviour)
     OP4->OP3->|
          OP2->|->OP1->out

'''
def fm_ablbrass_synth(pitch, ol, fr, sample_rate,max_ol,use_safe_cumsum=False):

    if(use_safe_cumsum==True):
        omega = cumsum_nd(2 * np.pi * pitch / sample_rate, 2*np.pi)
    else:
        omega = torch.cumsum(2 * np.pi * pitch / sample_rate, 1)

    # Torch unsqueeze with dim -1 adds a new dimension at the end of ol to match phases.

    op4_phase =  fr[OP4] * omega
    op4_output = torch.unsqueeze(ol[:,:,OP4], dim=-1) * torch.sin(op4_phase % (2*np.pi))

    op3_phase =  fr[OP3] * omega + 2 * np.pi * op4_output
    op3_output = torch.unsqueeze(ol[:,:,OP3], dim=-1) * torch.sin(op3_phase % (2*np.pi)) # output of stack of 2

    op2_phase =  fr[OP2] * omega
    op2_output = torch.unsqueeze(ol[:,:,OP2], dim=-1) * torch.sin(op2_phase % (2*np.pi)) # output stack of 1

    op1_phase =  fr[OP1] * omega + 2 * np.pi * (op2_output + op3_output)
    op1_output = torch.unsqueeze(ol[:,:,OP1], dim=-1) * torch.sin(op1_phase % (2*np.pi)) # global carrier

    return op1_output/max_ol

'''
String FM Synth - with phase wrapping (it does not change behaviour)
PATCH NAME: STRINGS 1
OP6->OP5->OP4->OP3 |
       (R)OP2->OP1 |->out
'''
def fm_string_synth(pitch, ol, fr, sample_rate,max_ol,use_safe_cumsum=False):

    if(use_safe_cumsum==True):
        omega = cumsum_nd(2 * np.pi * pitch / sample_rate, 2*np.pi)
    else:
        omega = torch.cumsum(2 * np.pi * pitch / sample_rate, 1)

    # Torch unsqueeze with dim -1 adds a new dimension at the end of ol to match phases.
    op6_phase =  fr[OP6] * omega
    op6_output = torch.unsqueeze(ol[:,:,OP6], dim=-1) * torch.sin(op6_phase % (2*np.pi))

    op5_phase =  fr[OP5] * omega + 2 * np.pi * op6_output
    op5_output = torch.unsqueeze(ol[:,:,OP5], dim=-1)*torch.sin(op5_phase % (2*np.pi))

    op4_phase =  fr[OP4] * omega + 2 * np.pi * op5_output
    op4_output = torch.unsqueeze(ol[:,:,OP4], dim=-1) * torch.sin(op4_phase % (2*np.pi))

    op3_phase =  fr[OP3] * omega + 2 * np.pi * op4_output
    op3_output = torch.unsqueeze(ol[:,:,OP3], dim=-1) * torch.sin(op3_phase % (2*np.pi)) # output of stack of 4

    op2_phase =  fr[OP2] * omega
    op2_output = torch.unsqueeze(ol[:,:,OP2], dim=-1) * torch.sin(op2_phase % (2*np.pi))

    op1_phase =  fr[OP1] * omega + 2 * np.pi * op2_output
    op1_output = torch.unsqueeze(ol[:,:,OP1], dim=-1) * torch.sin(op1_phase % (2*np.pi)) # output of stack of 2

    return (op3_output + op1_output)/max_ol

'''
Flute FM Synth - with phase wrapping (it does not change behaviour)
PATCH NAME: FLUTE 1
(R)OP6->OP5->|
   OP4->OP3->|
        OP2->|->OP1->out
'''
def fm_flute_synth(pitch, ol, fr, sample_rate,max_ol,use_safe_cumsum=False):

    if(use_safe_cumsum==True):
        omega = cumsum_nd(2 * np.pi * pitch / sample_rate, 2*np.pi)
    else:
        omega = torch.cumsum(2 * np.pi * pitch / sample_rate, 1)

    # Torch unsqueeze with dim -1 adds a new dimension at the end of ol to match phases.
    op6_phase =  fr[OP6] * omega
    op6_output = torch.unsqueeze(ol[:,:,OP6], dim=-1) * torch.sin(op6_phase % (2*np.pi))

    op5_phase =  fr[OP5] * omega + 2 * np.pi * op6_output
    op5_output = torch.unsqueeze(ol[:,:,OP5], dim=-1)*torch.sin(op5_phase % (2*np.pi)) # output of stack of 2

    op4_phase =  fr[OP4] * omega
    op4_output = torch.unsqueeze(ol[:,:,OP4], dim=-1) * torch.sin(op4_phase % (2*np.pi))

    op3_phase =  fr[OP3] * omega + 2 * np.pi * op4_output
    op3_output = torch.unsqueeze(ol[:,:,OP3], dim=-1) * torch.sin(op3_phase % (2*np.pi)) # output of stack of 2

    op2_phase =  fr[OP2] * omega
    op2_output = torch.unsqueeze(ol[:,:,OP2], dim=-1) * torch.sin(op2_phase % (2*np.pi)) # output stack of 1

    op1_phase =  fr[OP1] * omega + 2 * np.pi * (op2_output + op3_output + op5_output)
    op1_output = torch.unsqueeze(ol[:,:,OP1], dim=-1) * torch.sin(op1_phase % (2*np.pi)) # carrier

    return op1_output/max_ol

'''
Brass FM Synth - with phase wrapping (it does not change behaviour)
PATCH NAME: BRASS 3
OP6->OP5->OP4->|
       (R)OP3->|
          OP2->|->OP1->out
'''
def fm_brass_synth(pitch, ol, fr, sample_rate,max_ol,use_safe_cumsum=False):

    if(use_safe_cumsum==True):
        omega = cumsum_nd(2 * np.pi * pitch / sample_rate, 2*np.pi)
    else:
        omega = torch.cumsum(2 * np.pi * pitch / sample_rate, 1)

    # Torch unsqueeze with dim -1 adds a new dimension at the end of ol to match phases.
    op6_phase =  fr[OP6] * omega
    op6_output = torch.unsqueeze(ol[:,:,OP6], dim=-1) * torch.sin(op6_phase % (2*np.pi))

    op5_phase =  fr[OP5] * omega + 2 * np.pi * op6_output
    op5_output = torch.unsqueeze(ol[:,:,OP5], dim=-1)*torch.sin(op5_phase % (2*np.pi))

    op4_phase =  fr[OP4] * omega + 2 * np.pi * op5_output
    op4_output = torch.unsqueeze(ol[:,:,OP4], dim=-1) * torch.sin(op4_phase % (2*np.pi)) # output of stack of 3

    op3_phase =  fr[OP3] * omega
    op3_output = torch.unsqueeze(ol[:,:,OP3], dim=-1) * torch.sin(op3_phase % (2*np.pi)) # output of stack of 1

    op2_phase =  fr[OP2] * omega
    op2_output = torch.unsqueeze(ol[:,:,OP2], dim=-1) * torch.sin(op2_phase % (2*np.pi)) # output stack of 1

    op1_phase =  fr[OP1] * omega + 2 * np.pi * (op2_output + op3_output + op4_output)
    op1_output = torch.unsqueeze(ol[:,:,OP1], dim=-1) * torch.sin(op1_phase % (2*np.pi)) # carrier

    return op1_output/max_ol

@torch.no_grad()
def cumsum_nd(in_tensor,wrap_value=None):
    '''
    cumsum_nd() : cummulative sum - non differentiable and with wrap value.

    The problem with cumsum: when we work with phase tensors that are too large
    (i.e. more than a few tenths of seconds) cumsum gets to accumulate steps
    over a very large window, and it seems the float point variable loses precision.

    This workaround computes the accumulation step by step, resetting the
    accumulator in order for it to avoid to lose precision.

    NOTE: This implementation is very slow, and can't be used during training,
    only for final audio rendering on the test set.

    Assumes a tensor format used for audio rendering. [batch,len,1]

    NOTE:  Non integer frequency ratios do not work using current synthesis approach,
    because we render a common phase (wrapped using cumsum_nd) and then we multiply it
    by the frequency ratio. This introduces a misalignment if we multiply the wrapped phase
    by a non-integer frequency ratio.

    TODO: implement an efficient vectorial cumsum with wrapping we can use to accumulate
          phases from all oscillators separately
    '''
    print("[WARNING] Using non differentiable cumsum. Non-integer frequency ratios wont render well.")
    input_len = in_tensor.size()[1]
    nb = in_tensor.size()[0]
    acc = torch.zeros([nb,1,1])
    out_tensor = torch.zeros([nb,input_len,1])
    #print("in size{} - out size{}".format(in_tensor.size(),out_tensor.size()))
    for i in range(input_len):
        acc += in_tensor[:,i,0]
        if(wrap_value is not None):
            acc = acc - (acc > wrap_value)*wrap_value
        out_tensor[:,i,0] = acc
    return out_tensor
