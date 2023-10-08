import torch
import torch.nn as nn
from src.models.modules import get_mlp
from src.models.modules import TCN_block

class TCNdecoder(nn.Module):
    ''' TCN-Based decoder with sigmoid output    
    
    Parameters
    ----------
    in_channels : int
        number of channels output from mlp_input. ATTENTION: TCN in_channels = in_channels * n_input_feature
    hidden_channels : int
        number of hidden channels for every TCN_Block
    out_channels : int
        number of channels input to out_mlp.
    n_blocks : int
        number of TCN blocks
    stride : int
        convolution stride
    dilation_base : int
        base dilation for TCN_Block
    apply_padding : bool
    deploy_residual : bool
    input_keys : [str]
        input features
    input_sizes : [int]
        sizes of the input features
    output_keys : [str]
        output features
    output_sizes : [int]
        sizes of the output features
    '''
    def __init__(self,
                 in_channels = 128,
                 hidden_channels=64,
                 out_channels=256,
                 n_blocks=2,
                 kernel_size=3,
                 stride=1,
                 dilation_base=2,
                 apply_padding=True,
                 deploy_residual=False,
                 input_keys=None,
                 input_sizes=[1,1,16],
                 output_keys=['amplitude','harmonic_distribution','noise_bands'],
                 output_sizes=[1,100,65]
                 ):
        
        super().__init__()

        # Store receptive field
        dilation_factor = (dilation_base**n_blocks-1)/(dilation_base-1)
        self.receptive_field = 1 + 2*(kernel_size-1)*dilation_factor
        print("[INFO] TCNdecoder - receptive field is: {}".format(self.receptive_field))

        self.input_keys = input_keys
        self.ouput_keys = output_keys
        self.output_sizes = output_sizes
        
        n_keys = len(input_keys)
        
        # Generate MLPs of size: in_size: [1,1,16] ; n_layers = 3 (with layer normalization and leaky relu)
        if(n_keys == 2):
            self.in_mlps = nn.ModuleList([get_mlp(input_sizes[0], in_channels, 3),
                                          get_mlp(input_sizes[1], in_channels, 3)])
        elif(n_keys == 3):
            self.in_mlps = nn.ModuleList([get_mlp(input_sizes[0], in_channels, 3),
                                          get_mlp(input_sizes[1], in_channels, 3),
                                          get_mlp(input_sizes[2], in_channels, 3)])
        else:
            raise ValueError("Expected 2 or 3 input keys. got: {}".format(input_keys))

        # Generate TCN
        self.tcn = get_tcn(n_keys*in_channels, hidden_channels, out_channels, kernel_size, 
                         stride, dilation_base, apply_padding, deploy_residual, 
                         n_blocks)
        
        #Generate output MLP: in_size: out_channels + 2 ; n_layers = 3
        self.out_mlp = get_mlp(out_channels+2, out_channels, 3)
    
        # Projection matrix to generate control signals
        self.proj_matrices = []
        self.output_keys = output_keys
        self.output_sizes = output_sizes
        
        for v,k in enumerate(output_keys):
            self.proj_matrices.append(nn.Linear(out_channels, output_sizes[v]))

        self.proj_matrices = nn.ModuleList(self.proj_matrices)
        

    def forward(self,x):
         # Run through input_keys and load inputs accordingly
        hidden = torch.cat([self.in_mlps[v](x[k]) for v,k in enumerate(self.input_keys)],-1)
        
        # Reshape features to follow Conv1d convention (nb,ch,seq_Len)
        hidden = hidden.permute([0,-1,-2])
        hidden = self.tcn(hidden)
        hidden = hidden.permute([0,-1,-2])
        
        # Then, concatenate the embedding with the disentangled parameters of pitch and loudness (out_channels+2 size vector)
        hidden = torch.cat([hidden, x['f0_scaled'], x['loudness_scaled']], -1)
        # Run the embedding through the output MLP to obtain a 512-sized output vector.
        hidden = self.out_mlp(hidden)

        # Run embedding through a projection_matrix to get outputs
        controls = {}
        for v,k in enumerate(self.output_keys):
            controls[k] = self.proj_matrices[v](hidden)

        controls['f0_hz'] = x['f0']

        return controls
    


# TCN generator 
def get_tcn(in_channels, hidden_channels, out_channels, kernel_size, stride=1, dilation_base=1, apply_padding=True, deploy_residual=False, n_blocks=2):
        base = 0
        net = []

        net.append(TCN_block(in_channels, hidden_channels, hidden_channels, 
                             kernel_size, stride, dilation_base**base,
                             apply_padding, last_block=False, deploy_residual=deploy_residual))
        if(n_blocks>2):
            for i in range(n_blocks-2):
                base += 1
                net.append(TCN_block(hidden_channels,hidden_channels,hidden_channels, 
                             kernel_size, stride, dilation_base**base,
                             apply_padding, last_block=False, deploy_residual=deploy_residual))

        base += 1
        net.append(TCN_block(hidden_channels, hidden_channels, out_channels, 
                             kernel_size, stride, dilation_base**base,
                             apply_padding, last_block=True, deploy_residual=deploy_residual))

        return nn.Sequential(*net)
 