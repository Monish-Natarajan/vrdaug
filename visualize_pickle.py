import pickle
import os.path as osp

# see first few contents of pickle file
with open(osp.join('/home/monish/Flinders/code/vrdaug/data/vrd/train.pkl'), 'rb') as fid:
    u = pickle._Unpickler(fid)
    u.encoding = 'latin1'
    anno = u.load()

# format and print
# print(anno.keys())
print(anno[0])