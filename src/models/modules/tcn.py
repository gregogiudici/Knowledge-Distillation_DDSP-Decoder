import torch
import torch.nn as nn
import torch.nn.functional as F


class TCN_block(nn.Module):
    '''
    TCN Block
    '''
    def __init__(self,in_channels,hidden_channels,out_channels,
                kernel_size,stride=1,dilation=1,apply_padding=True,
                last_block=False,deploy_residual=False):
        super().__init__()
        block = []
        cnv1 = CausalConv1d(in_channels,hidden_channels,kernel_size,
            stride=stride,dilation=dilation,apply_padding=apply_padding)
        block.append(torch.nn.utils.weight_norm( cnv1 ) )
        block.append(nn.ReLU())
        block.append(nn.Dropout())

        cnv2 = CausalConv1d(hidden_channels,out_channels,kernel_size,
            stride=stride,dilation=dilation,apply_padding=apply_padding)
        block.append(torch.nn.utils.weight_norm( cnv2 ) )
        if(last_block == False):
            block.append(nn.ReLU())
            block.append(nn.Dropout())
          
        # weight_norm() causes error with deepcopy() so we need to detach the weights of the CausalConv1(s)    
        block[0].weight = block[0].weight.detach() # cnv1
        block[3].weight = block[3].weight.detach() # cnv2
        
        self.block = nn.Sequential(*block)
        #self.block[0].weight = self.block[0].weight.detach() # cnv1
        #self.block[3].weight = self.block[3].weight.detach() # cnv2
        
        self.residual = None
        if(deploy_residual):
            if(apply_padding):
                self.residual = nn.Conv1d(in_channels,out_channels,1,padding = 0,stride=stride)
            else:
                raise ValueError("Residual connection is only possible when padding is enabled.")

    def forward(self,data):
        block_out = self.block(data)
        if(self.residual is not None):
            residual = self.residual(data)
            block_out = block_out + residual
        return block_out


class CausalConv1d(torch.nn.Conv1d):
    '''
    Basic layer for implementing a TCN
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 apply_padding=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.apply_padding = apply_padding
        self.__padding = dilation*(kernel_size - 1)

    def forward(self, input):
        # Apply left padding using torch.nn.functional and then compute conv.
        if(self.apply_padding):
            return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))
        else:
            return super(CausalConv1d, self).forward(input)
