import sys
import torch 
import torch.utils.data as data
import torch.nn as nn
import tables
import json
import random
import numpy as np
import pickle
from utils import PAD_ID, SOS_ID, EOS_ID, UNK_ID, indexes2sent

    
class CodeSearchDataset(data.Dataset):
    """
    Dataset that has only positive samples.
    """

    def __init__(self,data_dir,code_vec,desc_vec):
        print("loading data...")
        table_code = tables.open_file(data_dir + code_vec)
        self.code = np.array(table_code.root.vecs)
        table_desc = tables.open_file(data_dir + desc_vec)
        self.desc = np.array(table_desc.root.vecs)
        self.data_len = self.code.shape[0]
        print("{} entries".format(self.data_len))


    
    def __getitem__(self, offset):
        code = self.code[offset]
        desc = self.desc[offset]
        return code,desc

        
    def __len__(self):
        return self.data_len
    


def load_vecs(fin):         
    """read vectors (2D numpy array) from a hdf5 file"""
    h5f = tables.open_file(fin)
    h5vecs= h5f.root.vecs
    
    vecs=np.zeros(shape=h5vecs.shape,dtype=h5vecs.dtype)
    vecs[:]=h5vecs[:]
    h5f.close()
    return vecs
        
def save_vecs(vecs, fout):
    fvec = tables.open_file(fout, 'w')
    atom = tables.Atom.from_dtype(vecs.dtype)
    filters = tables.Filters(complib='blosc', complevel=5)
    ds = fvec.create_carray(fvec.root,'vecs', atom, vecs.shape,filters=filters)
    ds[:] = vecs
    print('done')
    fvec.close()