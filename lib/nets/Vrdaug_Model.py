import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import os.path as osp
from lib.network import Conv2d, FC
import lib.network as network

import torchvision.ops as ops
import torchvision.models as models

import torchvision.ops.roi_pool as roi_pool 

class VGG_backbone(nn.Module):
    def __init__(self, args, bn=False):
        super(VGG_backbone, self).__init__()
        self.conv1 = nn.Sequential(Conv2d(3, 64, 3, same_padding=True, bn=bn),
                                   Conv2d(64, 64, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(Conv2d(64, 128, 3, same_padding=True, bn=bn),
                                   Conv2d(128, 128, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        network.set_trainable(self.conv1, requires_grad=False)
        network.set_trainable(self.conv2, requires_grad=False)

        self.conv3 = nn.Sequential(Conv2d(128, 256, 3, same_padding=True, bn=bn),
                                   Conv2d(256, 256, 3, same_padding=True, bn=bn),
                                   Conv2d(256, 256, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(Conv2d(256, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv5 = nn.Sequential(Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn))
        network.set_trainable(self.conv3, requires_grad=False)
        network.set_trainable(self.conv4, requires_grad=False)
        network.set_trainable(self.conv5, requires_grad=False)

    def forward(self,im_data):
        im_data = network.np_to_variable(im_data, is_cuda=False)
        im_data = im_data.permute(0, 3, 1, 2)

        x = self.conv1(im_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class Vrd_Model(nn.Module):
    def __init__(self, args, bn=False):
        super(Vrd_Model, self).__init__()

        self.n_rel = args.num_relations
        self.n_obj = args.num_classes

        # self.conv1 = nn.Sequential(Conv2d(3, 64, 3, same_padding=True, bn=bn),
        #                            Conv2d(64, 64, 3, same_padding=True, bn=bn),
        #                            nn.MaxPool2d(2))
        # self.conv2 = nn.Sequential(Conv2d(64, 128, 3, same_padding=True, bn=bn),
        #                            Conv2d(128, 128, 3, same_padding=True, bn=bn),
        #                            nn.MaxPool2d(2))
        # network.set_trainable(self.conv1, requires_grad=False)
        # network.set_trainable(self.conv2, requires_grad=False)

        # self.conv3 = nn.Sequential(Conv2d(128, 256, 3, same_padding=True, bn=bn),
        #                            Conv2d(256, 256, 3, same_padding=True, bn=bn),
        #                            Conv2d(256, 256, 3, same_padding=True, bn=bn),
        #                            nn.MaxPool2d(2))
        # self.conv4 = nn.Sequential(Conv2d(256, 512, 3, same_padding=True, bn=bn),
        #                            Conv2d(512, 512, 3, same_padding=True, bn=bn),
        #                            Conv2d(512, 512, 3, same_padding=True, bn=bn),
        #                            nn.MaxPool2d(2))
        # self.conv5 = nn.Sequential(Conv2d(512, 512, 3, same_padding=True, bn=bn),
        #                            Conv2d(512, 512, 3, same_padding=True, bn=bn),
        #                            Conv2d(512, 512, 3, same_padding=True, bn=bn))
        # network.set_trainable(self.conv3, requires_grad=False)
        # network.set_trainable(self.conv4, requires_grad=False)
        # network.set_trainable(self.conv5, requires_grad=False)
        
        self.vgg_backbone = VGG_backbone(args, bn=bn)

        self.roi_pool_size = (7,7)
        self.fc6 = FC(512 * 7 * 7, 4096)
        self.fc7 = FC(4096, 4096)        
        self.fc_obj = FC(4096, self.n_obj, relu=False)
        network.set_trainable(self.fc6, requires_grad=False)
        network.set_trainable(self.fc7, requires_grad=False)
        network.set_trainable(self.fc_obj, requires_grad=False)
        self.fc8 = FC(4096, 256)

        n_fusion = 256
        if(args.use_so):
            self.fc_so = FC(256*2, 256)
            n_fusion += 256

        if(args.loc_type == 1):
            self.fc_lov = FC(8, 256)
            n_fusion += 256
        elif(args.loc_type == 2):
            self.conv_lo = nn.Sequential(Conv2d(2, 96, 5, same_padding=True, stride=2, bn=bn),
                                       Conv2d(96, 128, 5, same_padding=True, stride=2, bn=bn),
                                       Conv2d(128, 64, 8, same_padding=False, bn=bn))
            self.fc_lov = FC(64, 256)
            n_fusion += 256

        if(args.use_obj):
            ####################################################
            self.emb = nn.Embedding(self.n_obj, 300)
            ####################################################
            network.set_trainable(self.emb, requires_grad=False)
            self.fc_so_emb = FC(300*2, 256)
            n_fusion += 256

        self.fc_fusion = FC(n_fusion, 256)
        self.fc_rel = FC(256, self.n_rel, relu=False)

    def forward(self, im_data, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, args):
        # im_data = network.np_to_variable(im_data, is_cuda=False)
        # im_data = im_data.permute(0, 3, 1, 2)
        boxes = network.np_to_variable(boxes, is_cuda=False)
        rel_boxes = network.np_to_variable(rel_boxes, is_cuda=False)
        SpatialFea = network.np_to_variable(SpatialFea, is_cuda=False)
        classes = network.np_to_variable(classes, is_cuda=False, dtype=torch.LongTensor)
        ix1 = network.np_to_variable(ix1, is_cuda=False, dtype=torch.LongTensor)
        ix2 = network.np_to_variable(ix2, is_cuda=False, dtype=torch.LongTensor)

        # x = self.conv1(im_data)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)

        x = self.vgg_backbone(im_data)

        ####################################################

        # print device and type of x, boxes, rel_boxes
        # print("x: ", x.device, x.type(), x.shape)
        # print("boxes: ", boxes.device, boxes.type(), boxes.shape)
        # print("rel_boxes: ", rel_boxes.device, rel_boxes.type(), rel_boxes.shape)
        x_so = roi_pool(x, boxes, self.roi_pool_size)

        # print("self.num_relations: ", self.n_rel)
        # print("self.num_classes: ", self.n_obj)

        num_candi_rels = rel_boxes.shape[0]

        # generate x_so randomly
        # x_so = torch.randn(num_candi_rels, 512, 7, 7)

        ####################################################
        
        x_so = x_so.view(x_so.size()[0], -1)
        x_so = self.fc6(x_so)
        x_so = F.dropout(x_so, training=self.training)
        x_so = self.fc7(x_so)
        x_so = F.dropout(x_so, training=self.training)
        
        obj_score = self.fc_obj(x_so)
        x_so = self.fc8(x_so)

        ####################################################

        x_u = roi_pool(x, rel_boxes, self.roi_pool_size)
        # generate x_u randomly
        # x_u = torch.randn(num_candi_rels, 512, 7, 7)
        
        ####################################################

        x = x_u.view(x_u.size()[0], -1)
        x = self.fc6(x)
        x = F.dropout(x, training=self.training)
        x = self.fc7(x)
        x = F.dropout(x, training=self.training)
        x = self.fc8(x)
        
        # Use Subject and Object Visual Embeddings in Addition to 
        # Visual Embeddings of the Union Region
        if(args.use_so):
            x_s = torch.index_select(x_so, 0, ix1)
            x_o = torch.index_select(x_so, 0, ix2)
            x_so = torch.cat((x_s, x_o), 1)
            x_so = self.fc_so(x_so)
            x = torch.cat((x, x_so), 1)

        # Use Spatial Embeddings
        if(args.loc_type == 1):
            lo = self.fc_lov(SpatialFea)
            x = torch.cat((x, lo), 1)            
        elif(args.loc_type == 2):
            lo = self.conv_lo(SpatialFea)
            lo = lo.view(lo.size()[0], -1)
            lo = self.fc_lov(lo)
            x = torch.cat((x, lo), 1)

        if(args.use_obj):
            ####################################################
            emb = self.emb(classes)
            # print("emb: ", emb.device, emb.type(), emb.shape)
            ####################################################
            
            emb = torch.squeeze(emb, 1)
            emb_s = torch.index_select(emb, 0, ix1)
            emb_o = torch.index_select(emb, 0, ix2)
            emb_so = torch.cat((emb_s, emb_o), 1)
            emb_so = self.fc_so_emb(emb_so)
            x = torch.cat((x, emb_so), 1)

        obj_score = 0

        x = self.fc_fusion(x)
        rel_score = self.fc_rel(x)
        return obj_score, rel_score


if __name__ == '__main__':
    m = Vrd_Model()