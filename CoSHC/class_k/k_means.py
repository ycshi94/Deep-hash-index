#coding=utf-8
from sklearn.cluster import KMeans
import tables
import numpy as np
import time

def save_vecs(vecs, fout):
    fvec = tables.open_file(fout, 'w')
    atom = tables.Atom.from_dtype(vecs.dtype)
    filters = tables.Filters(complib='blosc', complevel=5)
    ds = fvec.create_carray(fvec.root,'vecs', atom, vecs.shape,filters=filters)
    ds[:] = vecs
    print('done')
    fvec.close()
def load_h5(data_dir):
    print("loading data...")
    table_code = tables.open_file(data_dir)
    vec = np.array(table_code.root.vecs)
    return vec
# 生成20个数据
code_vec = load_h5('../data/github/code_train.h5')
desc_vec = load_h5('../data/github/desc_train.h5')
print(code_vec.shape)
half=len(code_vec)
data=np.vstack((code_vec,desc_vec))
print(data.shape)
# 调用K-mean函数库
itr=time.time()
est = KMeans(n_clusters=10)
est.fit(data)
print(time.time()-itr)

print("样本的类别所属标签", est.labels_)
print(est.labels_.shape)
print("聚类中心位置", est.cluster_centers_)
print(est.cluster_centers_.shape)
data_path="../data/github/"
save_vecs(np.vstack(est.labels_[:half]), data_path+'code_label.h5')
save_vecs(np.vstack(est.labels_[half:]), data_path+'desc_label.h5')
save_vecs(np.vstack(est.cluster_centers_), data_path+'center.h5')







