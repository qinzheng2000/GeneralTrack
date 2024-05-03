from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)

import os
import time

def write_results_bdd(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},{c},{l},-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores, classes, low in results:
            for tlwh, track_id, score, cls, low in zip(tlwhs, track_ids, scores, classes, low):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2), c=int(cls), l=int(low))
                f.write(line)
    logger.info('save results to {}'.format(filename))

def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))

def write_results_detection(filename, results):
    save_format = '{frame},-1,{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, scores in results:
            for tlwh, score in zip(tlwhs, scores):
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))

class MOTEvaluator:
    def __init__(
        self, args, dataloader, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.args = args

    def evaluate_bdd100k(
            self,
            model,
            distributed=False,
            half=False,
            trt_file=None,
            decoder=None,
            test_size=None,
            result_folder=None
    ):
        from yolox.tracker.byte_tracker_bdd import BYTETracker
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        # model = model.eval()
        # if half:
        #     model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        tracker = BYTETracker(self.args)
        ori_thresh = self.args.track_thresh
        for cur_iter, (imgs, _, info_imgs, ids, img_raw, det) in enumerate(
                progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]


                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = BYTETracker(self.args)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_bdd(result_filename, results)
                        results = []

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            # 取出detection信息，为了不影响推理速度，把他包含在json文件里比较好
            det = det.squeeze(0)
            # run tracking
            # if det is not None:
            # a = det.cpu().numpy()
            if len(det):
                online_targets = tracker.update(det, info_imgs, self.img_size, img_raw.cuda())
                online_tlwhs = []
                online_ids = []
                online_scores = []
                online_classes = []
                low = []
                for t in online_targets:
                    tlwh = t.observation
                    tid = t.track_id
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    online_classes.append(t.cls)
                    low.append(t.low)
                # save results
                results.append((frame_id, online_tlwhs, online_ids, online_scores, online_classes, low))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end

            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_bdd(result_filename, results)
