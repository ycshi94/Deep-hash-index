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
import faiss
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
from tensorboardX import SummaryWriter
import torch
import models.jointemb, configs, data_loader
from modules import get_cosine_schedule_with_warmup
from data_loader import *
def getint(matrix):
    matrix[matrix == -1] = 0
    matrix = matrix.astype(int)
    arr_reshaped = matrix.reshape(-1, 8)
    packed_arr = np.packbits(arr_reshaped, axis=1)
    arr_uint8 = packed_arr.reshape(-1,128//8).astype('uint8')
    return arr_uint8

def load_h5(data_dir):
    print("loading data...")
    table_code = tables.open_file(data_dir)
    vec = np.array(table_code.root.vecs)
    return vec


def similarity(vec1, vec2):
    # return np.dot(vec1, vec2.T)[:, 0]
    # return np.linalg.norm(vec1 - vec2, 1, axis=1)
    # return torch.norm(vec1 - vec2, p=1, dim=1)
    return torch.mm(vec1, vec2.t())[:, 0]


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
    itr = time.time()
    valid_result = validate(model, 18596, config['sim_measure'])
    print(time.time() - itr)


def validate(model, pool_size, sim_measure):
    model.eval()
    device = next(model.parameters()).device
    code_vec = load_h5('./data/github/code_test.h5')
    desc_vec = load_h5('./data/github/desc_test.h5')
    logger.info("validating..")
    n_processed = len(code_vec)
    code_reprs, desc_reprs = np.vstack(code_vec), np.vstack(desc_vec)
    code_hashs, desc_hashs = [], []
    for i in tqdm(range(len(code_reprs))):
        code_batch = torch.Tensor(np.expand_dims(code_reprs[i], axis=0)).to(device)
        desc_batch = torch.Tensor(np.expand_dims(desc_reprs[i], axis=0)).to(device)
        with torch.no_grad():
            code_hash = torch.sign(
                model.code_encoding(*code_batch).to(device).cpu().data).numpy().astype(np.float32)
            desc_hash = torch.sign(
                model.desc_encoding(*desc_batch).to(device).cpu().data).numpy().astype(np.float32)

        code_hashs.append(code_hash)
        desc_hashs.append(desc_hash)
    sum_1, sum_5, sum_10, sum_mrr = [], [], [], []

    code_hashs, desc_hashs = np.vstack(code_hashs), np.vstack(desc_hashs)
    code_or, desc_or =code_reprs, desc_reprs
    code_vec = getint(code_hashs)
    desc_vec = getint(desc_hashs)
    itr_time = time.time()
    quantizer = faiss.IndexBinaryFlat(128)
    nlist = 300
    index = faiss.IndexBinaryIVF(quantizer, 128, nlist)
    index.nprobe = 50
    index.train(code_vec)
    index.add(code_vec)
    print('index_time:', time.time() - itr_time)
    rerank=[0,10,20,30,50,100]
    for rerank_num in rerank:
        itr_time = time.time()
        sum_1, sum_5, sum_10, sum_mrr = [], [], [], []
        if rerank_num==0:
            dis, predict = index.search(desc_vec, 10)
            for i in tqdm(range(len(predict))):
                predict_1, predict_5, predict_10 = [int(predict[i][0])], [int(m) for m in predict[i][0:5]], [int(m) for
                                                                                                             m in
                                                                                                             predict[i][
                                                                                                             0:10]]
                sum_1.append(1.0) if i in predict_1 else sum_1.append(0.0)
                sum_5.append(1.0) if i in predict_5 else sum_5.append(0.0)
                sum_10.append(1.0) if i in predict_10 else sum_10.append(0.0)
                predict_list = predict[i].tolist()
                try:
                    rank = predict_list.index(i)
                    sum_mrr.append(1 / float(rank + 1))
                except:
                    sum_mrr.append(0)
            print('search_time:', "[",rerank_num,']',time.time() - itr_time)
            print('MRR={}, R@1={}, R@5={}, R@10={}'.format(np.mean(sum_mrr), np.mean(sum_1), np.mean(sum_5),
                                                           np.mean(sum_10)))
            continue
        dis, predict = index.search(desc_vec, rerank_num)
        for i in tqdm(range(len(predict))):
            predictv_10 = []
            for m in range(rerank_num):
                predictv_10.append(code_or[predict[i][m]])
            predictv_10 = np.array(predictv_10).astype('float32')
            index2 = faiss.IndexFlatL2(768)
            index2.add(predictv_10)
            _, predict_10 = index2.search(desc_or[i:i + 1], 10)

            repredict = []
            for n in predict_10[0]:
                repredict.append(predict[i][n])

            repredict_1, repredict_5, repredict_10 = [int(repredict[0])], [int(m) for m in repredict[0:5]], [int(m) for
                                                                                                             m in
                                                                                                             repredict[
                                                                                                             0:10]]

            sum_1.append(1.0) if i in repredict_1 else sum_1.append(0.0)
            sum_5.append(1.0) if i in repredict_5 else sum_5.append(0.0)
            sum_10.append(1.0) if i in repredict_10 else sum_10.append(0.0)
            # print(i,repredict_1,repredict,predict[i]) if i in repredict_10 else sum_10.append(0.0)
            predict_list = repredict
            try:
                rank = predict_list.index(i)
                sum_mrr.append(1 / float(rank + 1))
            except:
                sum_mrr.append(0)
        print('search_time:', "[",rerank_num,']',time.time() - itr_time)
        print('MRR={}, R@1={}, R@5={}, R@10={}'.format(np.mean(sum_mrr), np.mean(sum_1), np.mean(sum_5), np.mean(sum_10)))


    #rerank_num = 20

    return {'MRR': np.mean(sum_mrr), 'R@1': np.mean(sum_1), 'R@5': np.mean(sum_5), 'R@10': np.mean(sum_10)}


def parse_args():
    parser = argparse.ArgumentParser("Train and Validate The Code Search (Embedding) Model")
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='JointEmbeder', help='model name')
    parser.add_argument('--dataset', type=str, default='github', help='name of dataset.java, python')
    parser.add_argument('--reload_from', type=int, default=170000, help='epoch to reload from')
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
