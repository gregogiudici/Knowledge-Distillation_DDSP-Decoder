import torch
import torch.nn as nn
from src.models.modules import S4, S4_Layers
from src.models.modules import S4D, S4D_Layers
from src.models.modules import get_mlp

class S4decoder(nn.Module):
    ''' S4-Based decoder   
    
    Parameters
    ----------
    d_model : int
        Number of independent SSM copies; controls the size of the model.    
    d_state : int
        State size (dimensionality of parameters A, B, C). Generally shouldn't need to be adjusted and doens't affect speed much.
    dropout : int
        Dropout in S4(D)_Block.
    n_blocks : int
        number of S4 blocks
    prenorm : bool
        pre-normalization in S4-Block. If False is post-normalization in S4-Block
    diag : bool
        if True is used simpler S4D_Block from src/core/modules/s4d.py
    lr : float
        Passing in a number (e.g. 0.001) sets attributes of SSM parameers (A, B, dt). A custom optimizer hook is needed to configure the optimizer to set the learning rates appropriately for these parameters.
    measure : 
        Options for initialization of (A, B). For NPLR mode, recommendations are "legs", "fout", "hippo" (combination of both). For Diag mode, recommendations are "diag-inv", "diag-lin", "diag-legs", and "diag" (combination of diag-inv and diag-lin)
    mode : 
        Which kernel algorithm to use. 'nplr' is the full S4 model; 'diag' is the simpler S4D; 'slow' is a dense version for testing
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
                 d_model,
                 d_state=64,
                 dropout=0.2, 
                 n_blocks=1,
                 prenorm = False,
                 diag=False,
                 lr=0.001,
                 measure="legs",
                 mode="nplr",
                 input_keys=None,
                 input_sizes=[1,1,16],
                 output_keys=['amplitude','harmonic_distribution','noise_bands'],
                 output_sizes=[1,100,65]):
        
        super().__init__()
        
        self.input_keys = input_keys
        self.input_sizes = input_sizes
        n_keys = len(input_keys)
        
        # Generate MLPs of size: in_size: [1,1,16] ; n_layers = 3 (with layer normalization and leaky relu)
        if(n_keys == 2):
            self.in_mlps = nn.ModuleList([get_mlp(input_sizes[0], d_model, 3),
                                          get_mlp(input_sizes[1], d_model, 3)])
        elif(n_keys == 3):
            self.in_mlps = nn.ModuleList([get_mlp(input_sizes[0], d_model, 3),
                                          get_mlp(input_sizes[1], d_model, 3),
                                          get_mlp(input_sizes[2], d_model, 3)])
        else:
            raise ValueError("Expected 2 or 3 input keys. got: {}".format(input_keys))

        # Generate S4
        self.s4=get_s4(n_keys*d_model, d_state, dropout, n_blocks, prenorm, diag, lr, measure, mode)
        
        #Generate output MLP: in_size: n_keys*d_model + 2 ; n_layers = 3
        self.out_mlp = get_mlp(n_keys*d_model + 2, n_keys*d_model, 3)

        # Prejection Matrix
        self.proj_matrices = []
        self.output_keys = output_keys
        self.output_sizes = output_sizes
        for v,k in enumerate(output_keys):
            self.proj_matrices.append(nn.Linear(n_keys*d_model, output_sizes[v]))

        self.proj_matrices = nn.ModuleList(self.proj_matrices)

    def forward(self, x):
        # Run pitch and loudness and z (if available) inputs through the respectives input MLPs.
        # Then, concatenate the outputs in a flat vector.

        # Run through input_keys and load inputs accordingly
        hidden = torch.cat([self.in_mlps[v](x[k]) for v,k in enumerate(self.input_keys)],-1)

        # Run the flattened vector through the S4 layers.
        # Then, concatenate the embedding with the disentangled parameters of pitch and loudness (vector size = n_keys*d_size+2)
        hidden = torch.cat([self.s4(hidden), x['f0_scaled'], x['loudness_scaled']], -1)
        # Run the embedding through the output MLP to obtain the output vector.
        hidden = self.out_mlp(hidden)

        # Run embedding through a projection_matrix to get outputs
        controls = {}
        for v,k in enumerate(self.output_keys):
            controls[k] = self.proj_matrices[v](hidden)

        controls['f0_hz'] = x['f0']

        return controls

 
# S4(D) generator    
def get_s4(d_model, d_state=64, dropout=0.0, n_blocks=1, prenorm=False, diag=False, lr=0.001, measure="hippo", mode="nlpr"):
    if(diag):
        net = S4D_Layers(d_model, d_state, n_blocks, dropout, prenorm, lr)
    else: 
        net = S4_Layers(d_model, d_state, n_blocks, dropout, prenorm, lr, measure, mode)
    
    return net
