# encoding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.test_ann = "test.json"

        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
        self.random_size = (18, 32)
        self.max_epoch = 8
        self.print_interval = 20
        self.eval_interval = 5
        self.test_conf = 0.1
        self.nmsthre = 0.7
        self.no_aug_epochs = 1
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1

    def get_eval_loader(self, args_, batch_size, is_distributed, testdev=False):
        from yolox.data import ValTransform
        from yolox.data.datasets.mot_bdd import MOTDataset

        if testdev:
            valdataset = MOTDataset(
                data_dir='/datasets/bdd100k/images/track',
                json_file=self.test_ann,
                img_size=self.test_size,
                name='test',
                preproc=ValTransform(
                    rgb_means=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                resize=args_.resize
            )
        else:
            valdataset = MOTDataset(
                data_dir='/datasets/bdd100k/images/track',
                json_file=self.val_ann,
                img_size=self.test_size,
                name='val',
                preproc=ValTransform(
                    rgb_means=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                resize=args_.resize
            )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader
