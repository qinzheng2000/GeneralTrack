import torch
import torch.nn as nn
import time

from core.update import Hierarchical_Relation_Aggregation
from core.extractor import Feature_Extractor
from core.corr import CorrBlock
from core.utils import coords_grid

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

class Point2InstanceRelation(nn.Module):
    def __init__(self, args):
        super(Point2InstanceRelation, self).__init__()
        self.args = args
        self.hidden_dim = hdim = 128

        self.fnet = Feature_Extractor(output_dim=256, dropout=args.dropout)
        self.predict_block_similarity = Hierarchical_Relation_Aggregation(self.args)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def init(self, img):
        N, C, H, W = img.shape
        coords = coords_grid(N, H // 8, W // 8, device=img.device)

        return coords

    def forward(self, image1, image2, detection1, detection2):
        time2 = time.time()
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()  # rgb归一化,不共享内存
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        detection1.detach()
        detection2.detach()
        detection1_ = (detection1.clone()).int().detach()
        detection1_[:, :, 2:4] = torch.max(torch.div(detection1_[:, :, 2:4] , 8), torch.tensor([1]).cuda())
        detection1_[:, :, 4:6] = torch.max(torch.div(detection1_[:, :, 4:6] , 8), torch.tensor([1]).cuda())


        # Feature Extraction
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])
        time3 = time.time()

        # Feature Relation Extraction & Multi-scale
        corr_fn = CorrBlock(fmap1.float(), fmap2.float(), num_levels=self.args.corr_levels, radius=self.args.corr_radius)

        # Point-region Relation
        corr = corr_fn(self.init(image1).detach())

        # Hierarchical Relation Aggregation
        time4 = time.time()
        with autocast(enabled=self.args.mixed_precision):
            classification_score, label_gt = self.predict_block_similarity(detection1_, corr, detection1, detection2)

            time5 = time.time()
            t = [time2, time3, time4, time5]
            return classification_score, label_gt
