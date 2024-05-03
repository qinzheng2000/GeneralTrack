import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
import  time

from core.utils import position_encoding

class ConvMLPHead(nn.Module):

    def __init__(self, in_channels, representation_size, args, v):
        super(ConvMLPHead, self).__init__()

        self.args = args
        self.convc= nn.Conv2d(in_channels, representation_size, v)
        self.fc6 = nn.Linear(representation_size, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        # self.pooling1 = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_score = nn.Linear(representation_size, 1)

    def forward(self, x):
        # x = x.flatten(start_dim=1)   # (batchsize, 3,3,132) --> (batchsize, 1188)
        x = self.convc(x)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))       # (batchsize, 3,3,132)
        x = F.relu(self.fc7(x))
        # x = self.pooling1(x.transpose(1,3)).flatten(start_dim=1)
        scores = self.cls_score(x)

        return scores

class CompressConv(nn.Module):
    def __init__(self, args):
        super(CompressConv, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv = nn.Conv2d(256, 128, 3, padding=1)

    def forward(self, corr):
        cor = F.relu(self.convc1(corr))         # 注意这一块要不要拼接motion偏移量
        cor = F.relu(self.convc2(cor))

        out = F.relu(self.conv(cor))
        return out

class Hierarchical_Relation_Aggregation(nn.Module):
    def __init__(self, args):
        super(Hierarchical_Relation_Aggregation, self).__init__()
        self.args = args
        self.encoder_similarity = CompressConv(args)

        self.head = ConvMLPHead(132,256, args, args.roialign_size[0])

    def forward(self, detection1_, corr, detection1, detection2):
        relation_features = self.encoder_similarity(corr)

        # adjust coordinates format
        t1 = time.time()
        d1 = (detection1.clone())
        d1[..., 2:4] -= d1[..., 4:6] / 2
        d1[...,4:6] += d1[...,2:4]
        d1_crop = (d1.clone())
        d2 = (detection2.clone())
        d2[..., 2:4] -= d2[..., 4:6] / 2
        d2[...,4:6] += d2[...,2:4]
        d1 = d1[..., :8]
        d2 = d2[..., :8]
        t2 = time.time()

        # Create index column for cropping
        index_col = torch.arange(detection1_.shape[0]).cuda().view(detection1_.shape[0], 1).expand(detection1_.shape[0], detection1_.shape[1])
        d1_crop[:, :, 1] = index_col
        d1_crop = d1_crop.reshape(detection1_.shape[0]*detection1_.shape[1] , -1)

        # Perform cropping using roi_align function to get relation features
        t3 = time.time()
        relation_features_crop = roi_align(relation_features, d1_crop[:,1:6], self.args.roialign_size, spatial_scale=self.args.spatial_scale, sampling_ratio=-1).unsqueeze(1)

        # Expand d1 and d2 to 4D tensors for position encoding
        t4 = time.time()
        d_1 = d1.unsqueeze(2).expand(detection1_.shape[0], detection1_.shape[1], detection1_.shape[1], 8)
        d_2 = d2.unsqueeze(1).expand(detection1_.shape[0], detection1_.shape[1], detection1_.shape[1], 8)
        d_ = torch.cat((d_1,d_2), dim=-1)
        # Compute position encoding
        pos = position_encoding(d_,self.args.roialign_size, self.args)

        d_ = d_.reshape(detection1_.shape[0] * detection1_.shape[1]*detection1_.shape[1], 16)
        pos = pos.reshape(detection1_.shape[0] * detection1_.shape[1], detection1_.shape[1], pos.shape[3], self.args.roialign_size[0], self.args.roialign_size[1])
        relation_features_crop = relation_features_crop.expand(detection1_.shape[0]*detection1_.shape[1],detection1_.shape[1], relation_features_crop.shape[2], self.args.roialign_size[0], self.args.roialign_size[1])

        # Concatenate relation features and position encoding
        data_list = torch.cat((relation_features_crop,pos),dim=2)

        # Create label list
        label_list =(d_[:, 1] == d_[:, 9] )

        # Reshape data and filter valid data
        data_list = data_list.reshape(detection1_.shape[0] * detection1_.shape[1]*detection1_.shape[1], data_list.shape[2], self.args.roialign_size[0], self.args.roialign_size[1])
        index_valid = (d_[:, 0] != -1) & (d_[:, 8] != -1)
        data_list = data_list[index_valid]
        label_list = label_list[index_valid].float()

        t5 = time.time()

        outputs = self.head(data_list)
        t6 = time.time()

        return outputs, label_list