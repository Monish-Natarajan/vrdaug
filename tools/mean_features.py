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

# from lib.model import train_net, test_pre_net
# from lib.utils import save_checkpoint
# from lib.model import test_rel_net

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
    # net = Vrd_Model(args)
    # network.weights_normal_init(net, dev=0.01)
    # pretrained_model = osp.join(args.data_dir,'VGG_imagenet.npy')   
    # network.load_pretrained_npy(net, pretrained_model)
    # print("Loaded Pre-Trained VGG Feature Extractor!")
    # net.cuda()

    # # test
    # net.eval()    

    # # for each object class, extract mean pixel features of bounding box region
    # # also extract mean bounding box features --> x_center,y_center, w, h
    # mean_pixel_features = {}
    # mean_bbox_features = {}
    # for i in range(args.num_classes):
    #     mean_pixel_features[i] = []
    #     mean_bbox_features[i] = []
    
    for step in tqdm(range(train_data_layer._num_instance)):    
        train_data = train_data_layer.forward()    
        if(train_data is None):
            continue
        
        image_blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, rel_labels, rel_so_prior = train_data
        # feat_map = net.vgg_backbone(image_blob)
        # print("feat_map.shape: ", feat_map.shape)

        print(boxes[0])

        # break

    #     #####################################
    #     obj_score, rel_score = net(image_blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, args)
    #     #####################################      