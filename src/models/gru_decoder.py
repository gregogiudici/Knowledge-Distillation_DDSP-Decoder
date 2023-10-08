import torch
import torch.nn as nn
from src.models.modules import get_mlp

class GRUdecoder(nn.Module):
    ''' GRU-Based decoder 
    
    Parameters
    ----------
    hidden_size : int
        number of features in the hidden state
    num_layers : int
        number of gru layers
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
                 hidden_size=512, 
                 num_layers=1,
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
            self.in_mlps = nn.ModuleList([get_mlp(input_sizes[0], hidden_size, 3),
                                          get_mlp(input_sizes[1], hidden_size, 3)])
        elif(n_keys == 3):
            self.in_mlps = nn.ModuleList([get_mlp(input_sizes[0], hidden_size, 3),
                                          get_mlp(input_sizes[1], hidden_size, 3),
                                          get_mlp(input_sizes[2], hidden_size, 3)])
        else:
            raise ValueError("Expected 2 or 3 input keys. got: {}".format(input_keys))

        #Generate GRU: input_size = n_keys * hidden_size ; num_layers = 1 (that's the default config)
        self.gru = get_gru(n_keys, hidden_size, num_layers)

        #Generate output MLP: in_size: hidden_size + 2 ; n_layers = 3
        self.out_mlp = get_mlp(hidden_size + 2, hidden_size, 3)

        # Projection matrix
        self.proj_matrices = []
        self.output_keys = output_keys
        self.output_sizes = output_sizes
        
        for v,k in enumerate(output_keys):
            self.proj_matrices.append(nn.Linear(hidden_size,output_sizes[v]))

        self.proj_matrices = nn.ModuleList(self.proj_matrices)

    def forward(self, x):
        # Run pitch and loudness and z (if available) inputs through the respectives input MLPs.
        # Then, concatenate the outputs in a flat vector.

        # Run through input_keys and load inputs accordingly
        hidden = torch.cat([self.in_mlps[v](x[k]) for v,k in enumerate(self.input_keys)],-1)

        # Run the flattened vector through the GRU.
        # The GRU predicts the embedding.
        # Then, concatenate the embedding with the disentangled parameters of pitch and loudness (nhid+2 size vector)
        hidden = torch.cat([self.gru(hidden)[0], x['f0_scaled'], x['loudness_scaled']], -1)
        # Run the embedding through the output MLP to obtain a 512-sized output vector.
        hidden = self.out_mlp(hidden)


        # Run embedding through a projection_matrix to get outputs
        controls = {}
        for v,k in enumerate(self.output_keys):
            controls[k] = self.proj_matrices[v](hidden)

        controls['f0_hz'] = x['f0']

        return controls
    

# GRU generator
def get_gru(n_input, hidden_size, num_layers):
    return nn.GRU(n_input * hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
