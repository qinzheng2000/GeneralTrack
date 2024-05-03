from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger
from yolox.evaluators import MOTEvaluator

import argparse
import os
import random
import warnings
import glob
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path
import os.path as osp

from configs.config_utils import Config

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument('--stage', default='BDD100K', help="determines which dataset to use for tracking")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=1, type=int, help="device for training"
    )
    parser.add_argument(
        "--local_rank", default=0, type=int, help="local rank for dist training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=True,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=True,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # det args
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    # tracking args
    parser.add_argument("--byte", default=False)
    parser.add_argument("--track_thresh", type=float, default=0, help="tracking confidence threshold")
    parser.add_argument("--det_thresh", type=float, default=0, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=0, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0, help="matching threshold for tracking")
    parser.add_argument("--matchlost_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument("--min_box_area", type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    # relation args
    parser.add_argument('--resize', type=int, nargs='+', default=(0, 0))
    parser.add_argument('--restore_ckpt', default=' ', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--spatial_scale', type=float, default=1/8)
    parser.add_argument('--roialign_size', default=(0, 0))  # （h, w）
    parser.add_argument('--corr_radius', type=int, default=0, help="原来[400, 720]都是4，[720, 1280]是7")
    parser.add_argument('--corr_levels', type=int, default=0)

    return parser



def load_config(filename:str = None):
    """load and print config"""
    print('loading config from ' + filename + ' ...')
    try:
        configfile = Config(filename=filename)
        config = configfile._cfg_dict
    except (FileNotFoundError, IOError):
        config = dict()
        print('warning: fail to load the config!')
    return config

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    cudnn.benchmark = True

    rank = args.local_rank
    # rank = get_local_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    results_folder = os.path.join(file_name, args.stage, "track_results")
    if args.test:
        results_folder = results_folder + 'test'
    os.makedirs(results_folder, exist_ok=True)



    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    #logger.info("Model Structure:\n{}".format(str(model)))

    val_loader = exp.get_eval_loader(args, args.batch_size, is_distributed, args.test)
    evaluator = MOTEvaluator(
        args=args,
        dataloader=val_loader,
        img_size=exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        num_classes=exp.num_classes,
        )

    torch.cuda.set_device(rank)
    # model.cuda(rank)
    # model.eval()

    # if not args.speed and not args.trt:
    #     if args.ckpt is None:
    #         ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
    #     else:
    #         ckpt_file = args.ckpt
    #     logger.info("loading checkpoint")
    #     loc = "cuda:{}".format(rank)
    #     ckpt = torch.load(ckpt_file, map_location=loc)
    #     # load the model state dict
    #     model.load_state_dict(ckpt["model"])
    #     logger.info("loaded checkpoint done.")
    #
    # if is_distributed:
    #     model = DDP(model, device_ids=[rank])
    #
    # if args.fuse:
    #     logger.info("\tFusing model...")
    #     model = fuse_model(model)
    #
    # if args.trt:
    #     assert (
    #         not args.fuse and not is_distributed and args.batch_size == 1
    #     ), "TensorRT model is not support model fusing and distributed inferencing!"
    #     trt_file = os.path.join(file_name, "model_trt.pth")
    #     assert os.path.exists(
    #         trt_file
    #     ), "TensorRT model is not found!\n Run tools/trt.py first!"
    #     model.head.decode_in_inference = False
    #     decoder = model.head.decode_outputs
    # else:
    #     trt_file = None
    #     decoder = None

    model = None
    decoder = None
    trt_file = None

    # start evaluate
    evaluator.evaluate_bdd100k(
        model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, results_folder
    )
    logger.info('Completed')


if __name__ == "__main__":
    args = make_parser().parse_args()
    cfg_path = osp.join('configs', f'{args.stage}.py')
    args.__dict__.update(load_config(cfg_path))

    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu),
    )
