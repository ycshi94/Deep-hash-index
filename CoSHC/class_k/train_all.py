#coding=utf-8
import torch.utils.data as data
import numpy as np
import logging
import math
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
from load_data import *
from models import *
from torch.optim.lr_scheduler import LambdaLR
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., 0.5 * (1. + math.cos(math.pi * float(num_cycles) * 2. * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
import time
import sys
def validate(model,train_set):
    model.eval()
    data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=307650,
                                              shuffle=True, drop_last=True, num_workers=1)
    acc=[]
    for batch in data_loader:
        code_vec = [tensor.to(device) for tensor in batch[:1]]
        desc_vec = [tensor.to(device) for tensor in batch[2:3]]
        codeout = model.getclass(code_vec[0])
        descout = model.getclass(desc_vec[0])
        code_prediction = torch.max(F.softmax(codeout,dim=1), dim=1)[1].cpu().data.numpy().squeeze()
        desc_prediction = torch.max(F.softmax(descout, dim=1), dim=1)[1].cpu().data.numpy().squeeze()
        accuracy = sum(code_prediction == desc_prediction) / len(code_prediction)  # 计算准确度
        acc.append(accuracy)
    return {'accuracy': np.mean(acc)}

if __name__ == "__main__":
    # x0，x1是数据，y0,y1是标签
    fh = logging.FileHandler(f"logs.txt")
    logger.addHandler(fh)
    train_set = load_Data()
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=64, shuffle=True, drop_last=True,
                                              num_workers=1)
    model = all_Net(n_feature=len(train_set.code_vec[0]), n_output=10)  # 几个类别就几个 output
    model.to(device)
    # 训练网络
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=5000,
        num_training_steps=len(data_loader) * 2001)  # do not foget to modify the number when dataset is changed
    n_iters = len(data_loader)
    itr_global = 1
    for epoch in range(int(0 / n_iters) + 1, 2001 + 1):
        itr_start_time = time.time()
        losses = []
        for batch in data_loader:
            model.train()
            code_vec = [tensor.to(device) for tensor in batch[:1]]
            code_label = [tensor.to(device) for tensor in batch[1:2]]
            desc_vec = [tensor.to(device) for tensor in batch[2:3]]
            desc_label = [tensor.to(device) for tensor in batch[3:]]
            loss = model(code_vec[0], code_label[0],desc_vec[0], desc_label[0])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            losses.append(loss.item())
            if itr_global % 100 == 0:
                elapsed = time.time() - itr_start_time
                logger.info('epo:[%d/%d] itr:[%d/%d] step_time:%ds Loss=%.5f' %
                            (epoch, 2001, itr_global % n_iters, n_iters, elapsed, np.mean(losses)))
                losses = []
                itr_start_time = time.time()
            itr_global = itr_global + 1
            if itr_global % 20000 == 0:
                logger.info("validating..")
                valid_result = validate(model, train_set)
                logger.info(valid_result)
            if itr_global % 10000 == 0:
                ckpt_path = f'./model/step{itr_global}.h5'
                save_model(model, ckpt_path)
            if itr_global == 200000:
                sys.exit(0)