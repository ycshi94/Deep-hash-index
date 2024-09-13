#coding=utf-8
from models import *
from load_data import *
import numpy as np
import torch.utils.data as data
import os
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
def load_model(model,step, to_device):
    path = f'step{step}.h5'
    assert os.path.exists(path), f'Code model not found'
    model.load_state_dict(torch.load(path, map_location=to_device))
def validate(code_model, valid_set):
    code_model.eval()
    #438111，48678，500000
    data_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=438111,
                                              shuffle=True, drop_last=True, num_workers=1)
    acc = []
    for batch in data_loader:
        code_vec = [tensor.to(device) for tensor in batch[:1]]
        desc_vec = [tensor.to(device) for tensor in batch[1:]]
        code_out = torch.max(F.softmax(code_model.getclass(code_vec[0]), dim=1), dim=1)[1].data.numpy().squeeze()
        desc_out = torch.max(F.softmax(code_model.getclass(desc_vec[0]), dim=1), dim=1)[1].data.numpy().squeeze()
        accuracy = sum(code_out == desc_out) / len(code_vec[0])  # 计算准确度
        acc.append(accuracy)
    print('accuracy',np.mean(acc))


if __name__ == '__main__':
    code_model=code_Net(n_feature=512, n_output=10)
    step=420000
    gpu_id=0
    valid_set=load_all_Data('test')
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    load_model(code_model,step, device)
    logger.info("validating..")
    valid_result = validate(code_model,valid_set)

