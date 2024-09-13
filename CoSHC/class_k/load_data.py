import tables
import torch.utils.data as data
import torch
def save_model(model, ckpt_path):
    torch.save(model.state_dict(), ckpt_path)
def load_class_h5(data_dir):
    print("loading data...")
    table_code = tables.open_file(data_dir)
    fin = torch.Tensor(table_code.root.vecs)
    return fin
class load_Data(data.Dataset):
    def __init__(self):
        self.code_vec = load_class_h5('../data/github/code_train.h5')
        self.code_label = torch.squeeze(load_class_h5('../data/github/code_label.h5'))
        self.desc_vec = load_class_h5('../data/github/desc_train.h5')
        self.desc_label = torch.squeeze(load_class_h5('../data/github/desc_label.h5'))
        self.data_len=len(self.desc_vec)
    def __getitem__(self, offset):
        code_vec = self.code_vec[offset]
        code_label = self.code_label[offset]
        desc_vec = self.desc_vec[offset]
        desc_label = self.desc_label[offset]
        return code_vec, code_label,desc_vec,desc_label
    def __len__(self):
        return self.data_len

class load_all_Data(data.Dataset):
    def __init__(self,type):
        self.desc_vec = load_class_h5(f'../data/github/desc_{type}.h5')
        self.code_vec = load_class_h5(f'../data/github/code_{type}.h5')
        self.data_len=len(self.desc_vec)
    def __getitem__(self, offset):
        code_vec=self.code_vec[offset]
        desc_vec = self.desc_vec[offset]
        return code_vec, desc_vec
    def __len__(self):
        return self.data_len