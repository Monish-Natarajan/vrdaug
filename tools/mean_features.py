import os
import os.path as osp
import sys
import pickle
import argparse
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.nn.init

import _init_paths
from lib.nets.Vrdaug_Model import Vrd_Model
import lib.network as network
from lib.data_layers.my_vrd_data_layer import VrdDataLayer

import numpy as np
from tqdm import tqdm

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='PyTorch VRD Training')
    parser.add_argument('--name', dest='name',
                        help='experiment name',
                        default=None, type=str)
    parser.add_argument('--dataset', dest='ds_name',
                        help='dataset name',
                        default='vrd', type=str)    
    parser.add_argument('--model_type', dest='model_type',
                        help='model type: RANK_IM, LOC',
                        default=None, type=str)
    parser.add_argument('--no_so', dest='use_so', action='store_false')
    parser.set_defaults(use_so=True)
    parser.add_argument('--no_obj', dest='use_obj', action='store_false')
    parser.set_defaults(use_obj=True)
    parser.add_argument('--no_obj_prior', dest='use_obj_prior', action='store_false')
    parser.set_defaults(use_obj_prior=True)        
    parser.add_argument('--loc_type', default=0, type=int)
    parser.add_argument('--epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--print-freq', '-p', default=1, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--data_dir', default='/content/drive/MyDrive/vrdaug/data', type=str, metavar='PATH',
                    help='path to data directory')
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    global args
    args = parse_args()
    print(args)
    
    # hyper-parameters
    args.proposal = osp.join(args.data_dir,'vrd/proposal.pkl')
    
    # load data
    train_data_layer = VrdDataLayer(args.ds_name, 'train', model_type = args.model_type, data_dir = args.data_dir)    
    args.num_relations = train_data_layer._num_relations
    args.num_classes = train_data_layer._num_classes
    
    # # load net
    net = Vrd_Model(args)
    # network.weights_normal_init(net, dev=0.01)
    # pretrained_model = osp.join(args.data_dir,'VGG_imagenet.npy')   
    # network.load_pretrained_npy(net, pretrained_model)
    # print("Loaded Pre-Trained VGG Feature Extractor!")
    # net.cuda()

    # # test
    net.eval()

    mean_class_feat = [torch.zeros(512,1,1)]*args.num_classes
    class_inst_count = [0]*args.num_classes    
    
    img_cnt=0
    for step in range(train_data_layer._num_instance):    
        train_data = train_data_layer.forward()    
        if(train_data is None):
            continue
        
        image_blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, rel_labels, rel_so_prior = train_data
        feat = net.vgg_backbone(image_blob) # image feature map --> torch.Size([1, 512, _, _])

        # merge ix1 and ix2
        uniq_inst = np.unique(np.concatenate([ix1, ix2], axis=0))
        for iid in uniq_inst:
            
            im_ht, im_width = image_blob.shape[2:]
            ft_ht, ft_width = feat.shape[2:]
            
            bbox = boxes[iid][1:5]
            newbbox = bbox.copy()
            newbbox[0] = (int) (bbox[0] * ft_width / im_width)
            newbbox[1] = (int) (bbox[1] * ft_ht / im_ht)
            newbbox[2] = (int) (bbox[2] * ft_width / im_width)
            newbbox[3] = (int) (bbox[3] * ft_ht / im_ht)

            inst_feat_map = feat[0][:, newbbox[1]:newbbox[3], newbbox[0]:newbbox[2]]
            mean_values = torch.mean(inst_feat_map, dim=(1, 2), keepdim=True)
            mean_class_feat[classes[iid]] += mean_values
            class_inst_count[classes[iid]] += 1

            print("inst feat shape: ", inst_feat_map.shape)
            img_cnt += 1
            print("img_cnt: ", img_cnt)
    
    for i in range(args.num_classes):
        mean_class_feat[i] /= class_inst_count[i]

    with open(osp.join(args.data_dir, 'mean_class_feat.pkl'), 'wb') as f:
        pickle.dump(mean_class_feat, f)