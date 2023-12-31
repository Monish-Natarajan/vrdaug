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

from lib.model import train_net, test_pre_net
from lib.utils import save_checkpoint
# from lib.model import test_rel_net

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
    # ------------    
    args.proposal = osp.join(args.data_dir,'vrd/proposal.pkl')
    
    # load data
    train_data_layer = VrdDataLayer(args.ds_name, 'train', model_type = args.model_type)    
    args.num_relations = train_data_layer._num_relations
    args.num_classes = train_data_layer._num_classes
    
    # load net
    net = Vrd_Model(args)
    network.weights_normal_init(net, dev=0.01)
    pretrained_model = osp.join(args.data_dir,'VGG_imagenet.npy')   
    network.load_pretrained_npy(net, pretrained_model)
    print("Loaded Pre-Trained VGG Feature Extractor!")
    
    # Initial object embedding with word2vec
    emb_path = osp.join(args.data_dir,'vrd/params_emb.pkl')
    with open(emb_path,'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        emb_init = u.load()
    net.state_dict()['emb.weight'].copy_(torch.from_numpy(emb_init))
    print("Loaded word2vec embeddings!")
    
    net.cuda()    
    params = list(net.parameters())
    momentum = 0.9
    weight_decay = 0.0005
    args.criterion = nn.MultiLabelMarginLoss().cuda()    
    opt_params = [{'params': net.fc8.parameters(), 'lr': args.lr*10},
                  {'params': net.fc_fusion.parameters(), 'lr': args.lr*10},
                  {'params': net.fc_rel.parameters(), 'lr': args.lr*10},                  
                  ]
    
    if(args.use_so):
        opt_params.append({'params': net.fc_so.parameters(), 'lr': args.lr*10})
    if(args.loc_type == 1):
        opt_params.append({'params': net.fc_lov.parameters(), 'lr': args.lr*10})
    elif(args.loc_type == 2):
        opt_params.append({'params': net.conv_lo.parameters(), 'lr': args.lr*10})
        opt_params.append({'params': net.fc_lov.parameters(), 'lr': args.lr*10})
    if(args.use_obj):
        opt_params.append({'params': net.fc_so_emb.parameters(), 'lr': args.lr*10})  

    args.optimizer = torch.optim.Adam(opt_params, lr=args.lr, weight_decay=weight_decay)
    
    # if args.resume:
    #     if osp.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']            
    #         net.load_state_dict(checkpoint['state_dict'])
    #         args.optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))    
    
    res_file = f'../experiment/results/{args.name}.txt'
    if not osp.exists('../experiment/results'):
        os.makedirs('../experiment/results')
    save_dir = f'../models/{args.name}'
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    
    # headers = ["Epoch","Pre R@50", "ZS", "R@100", "ZS", "Rel R@50", "ZS", "R@100", "ZS"]
    headers = ["Epoch","Pre R@50", "ZS", "R@100", "ZS"]
    res = []
    res.append((0,) + test_pre_net(net, args))
    with open(res_file, 'w') as f:
        f.write(tabulate(res, headers))
    
    for epoch in range(args.start_epoch, args.epochs):
        print('Epoch: {}'.format(epoch+1))
        train_net(train_data_layer, net, epoch, args)
        
        # res.append((epoch,) + test_pre_net(net, args)+test_rel_net(net, args))
        res.append((epoch,) + test_pre_net(net, args))

        with open(res_file, 'w') as f:
            f.write(tabulate(res, headers))
        save_checkpoint(f'{save_dir}/epoch_{epoch}_checkpoint.pth.tar', {
            'epoch': epoch,
            'state_dict': net.state_dict(),            
            'optimizer' : args.optimizer.state_dict(),
        })

        break
