from torch.utils.data import Dataset
import h5py
import numpy as np
import random
import torch
import math

class h5Dataset(Dataset):
    """ Class to load the h5 dataset pre-computed with dataset/prepare_data.py
    Parameters
    ----------
    sr : int
        audio sample rate
    data_path : str
        path of dataset location
    input_keys : [str]
        list of keys used in data dictionary
    max_audio_val : int
        maximum audio value
    device : str
        load data on cpu or gpu or ddp...
    ---------
    
    """
    def __init__(self, sr, data_path, input_keys, max_audio_val=1, device='cpu'):
        self.sr = sr
        self.data_path = data_path
        self.input_data_dicts,self.dataset_len = self.cache_data(self.data_path,len(input_keys))
        self.max_audio_val = max_audio_val
        self.input_keys = input_keys

    def cache_data(self, data_path,nfeatures):
        '''
        Load data to dictionary in RAM
        '''
        h5f = h5py.File(data_path, 'r')
        cache = {}
        keys = h5f.keys()
        nkeys = len(keys)
        ndata = (len(keys)//nfeatures)
        if((nkeys//nfeatures)*nfeatures != nkeys):
            raise Exception("Unexpected dataset len.")

        for key in keys:
            cache[key] = np.array(h5f[key])
        h5f.close()

        return cache, ndata

    def __getitem__(self, idx):
        #print("[DEBUG] __getitem__ fetching: {}".format(idx))

        #Generate current item keys to fetch from RAM cache
        item_keys = [f'{idx}_{k}' for k in self.input_keys ]

        # Load dictionary
        x = {}
        for v,k in enumerate(self.input_keys):
            x[k] = torch.tensor(self.input_data_dicts[item_keys[v]]).unsqueeze(-1)

        #for k in x.keys():
        #    print(f'{k}: {x[k].shape} ',end='')
        #print('')

        return x

    def __len__(self):
        return self.dataset_len
