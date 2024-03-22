import os
import sys
import random
import time
from datetime import datetime
import numpy as np
import math
import argparse

random.seed(42)
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
from tensorboardX import SummaryWriter
import torch
import models.jointemb, configs, data_loader
from modules import get_cosine_schedule_with_warmup
from utils import similarity, normalize
from data_loader import *


def eval_(args):
    fh = logging.FileHandler(f"./output/{args.model}/{args.dataset}/logseval.txt")
    logger.addHandler(fh)
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    tb_writer = SummaryWriter(f"./output/{args.model}/{args.dataset}/logseval/{timestamp}") if args.visual else None
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    config = getattr(configs, 'config_' + args.model)()
    if args.automl:
        config.update(vars(args))
    print(config)
    data_path = args.data_path + args.dataset + '/'
    valid_set = CodeSearchDataset(data_path, config['code_vec_test'], config['desc_vec_test'])

    logger.info('Constructing Model..')
    model = getattr(models.jointemb, args.model)(config)

    def save_model(model, ckpt_path):
        torch.save(model.state_dict(), ckpt_path)

    def load_model(model, ckpt_path, to_device):
        assert os.path.exists(ckpt_path), f'Weights not found'
        model.load_state_dict(torch.load(ckpt_path, map_location=to_device))

    if args.reload_from > 0:
        ckpt = f'./output/{args.model}/{args.dataset}/models/step{args.reload_from}.h5'
        load_model(model, ckpt, device)

    model.to(device)
    logger.info("validating..")
    valid_result = validate(valid_set, model, 18596, config['sim_measure'])


def validate(valid_set, model, pool_size, sim_measure):
    model.eval()
    device = next(model.parameters()).device
    data_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=18596,
                                              shuffle=False, drop_last=True, num_workers=1)
    accs, mrrs, maps, ndcgs = [], [], [], []
    code_reprs, desc_reprs = [], []
    n_processed = 0
    for batch in tqdm(data_loader):
        code_batch = [tensor.to(device) for tensor in batch[:1]]
        desc_batch = [tensor.to(device) for tensor in batch[1:2]]
        with torch.no_grad():
            code_repr = torch.sign(
                model.code_encoding(*code_batch).to(device).cpu().data).numpy().astype(np.float32)
            desc_repr = torch.sign(
                model.desc_encoding(*desc_batch).to(device).cpu().data).numpy().astype(np.float32)

        code_reprs.append(code_repr)
        desc_reprs.append(desc_repr)
        n_processed += batch[0].size(0)
    sum_1, sum_5, sum_10, sum_mrr = [], [], [], []
    code_reprs, desc_reprs = np.vstack(code_reprs), np.vstack(desc_reprs)
    print(code_reprs[1])
    data_path = './vec/'
    output_path_code = f"{data_path}code_test_hash.h5"
    save_vecs(code_reprs, output_path_code)
    output_path_desc = f"{data_path}desc_test_hash.h5"
    save_vecs(desc_reprs, output_path_desc)

def parse_args():
    parser = argparse.ArgumentParser("Train and Validate The Code Search (Embedding) Model")
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='JointEmbeder', help='model name')
    parser.add_argument('--dataset', type=str, default='github', help='name of dataset.java, python')
    parser.add_argument('--reload_from', type=int, default=310000, help='epoch to reload from')
    parser.add_argument('-g', '--gpu_id', type=int, default=1, help='GPU ID')
    parser.add_argument('-v', "--visual", action="store_true", default=False,
                        help="Visualize training status in tensorboard")
    parser.add_argument('--automl', action='store_true', default=False, help='use automl')
    parser.add_argument('--log_every', type=int, default=100, help='interval to log autoencoder training results')
    parser.add_argument('--valid_every', type=int, default=5000, help='interval to validation')
    parser.add_argument('--save_every', type=int, default=10000, help='interval to evaluation to concrete results')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--n_hidden', type=int, default=-1,
                        help='number of hidden dimension of code/desc representation')
    parser.add_argument('--lstm_dims', type=int, default=-1)
    parser.add_argument('--margin', type=float, default=-1)
    parser.add_argument('--sim_measure', type=str, default='cos', help='similarity measure for training')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--pause', default=0, type=int)
    parser.add_argument('--iteration', default=0, type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(f'./output/{args.model}/{args.dataset}/models', exist_ok=True)
    os.makedirs(f'./output/{args.model}/{args.dataset}/tmp_results', exist_ok=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    eval_(args)
