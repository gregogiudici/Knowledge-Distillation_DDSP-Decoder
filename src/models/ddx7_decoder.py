import torch
import torch.nn as nn
from src.models.modules import TCN_block

'''
TCN-Based decoder for DDX7
'''
class FMDecoder(nn.Module):
    '''
    FM Decoder with sigmoid output
    '''
    def __init__(self,
                 n_blocks=2,
                 hidden_channels=64,
                 out_channels=6,
                 kernel_size=3,
                 dilation_base=2,
                 apply_padding=True,
                 deploy_residual=False,
                 input_keys=None,
                 z_size=None,
                 output_complete_controls=True):
        super().__init__()

        # Store receptive field
        dilation_factor = (dilation_base**n_blocks-1)/(dilation_base-1)
        self.receptive_field = 1 + 2*(kernel_size-1)*dilation_factor
        print("[INFO] FMDecoder (TCN) - receptive field is: {}".format(self.receptive_field))

        self.input_keys = input_keys
        n_keys = len(input_keys)
        self.output_complete_controls = output_complete_controls

        if(n_keys == 2):
            in_channels = 2
        elif(n_keys == 3):
            in_channels = 2 + z_size
        else:
            raise ValueError("Expected 2 or 3 input keys. got: {}".format(input_keys))

        base = 0
        net = []

        net.append(TCN_block(in_channels,hidden_channels,hidden_channels,kernel_size,
            dilation=dilation_base**base,apply_padding=apply_padding,
            deploy_residual=deploy_residual))
        if(n_blocks>2):
            for i in range(n_blocks-2):
                base += 1
                net.append(TCN_block(hidden_channels,hidden_channels,hidden_channels,
                    kernel_size,dilation=dilation_base**base,apply_padding=apply_padding))

        base += 1
        net.append(TCN_block(hidden_channels,hidden_channels,out_channels,kernel_size,
            dilation=dilation_base**base,apply_padding=apply_padding,
            deploy_residual=deploy_residual,last_block=True))

        self.net = nn.Sequential(*net)

    def forward(self,x):
        # Reshape features to follow Conv1d convention (nb,ch,seq_Len)
        conditioning = torch.cat([x[k] for v,k in enumerate(self.input_keys)],-1).permute([0,-1,-2])

        ol = self.net(conditioning)
        ol = ol.permute([0,-1,-2])
        if self.output_complete_controls is True:
            synth_params = {
                'f0_hz': x['f0'], #In Hz
                'ol': ol
                }
        else:
            synth_params = ol
        return synth_params
