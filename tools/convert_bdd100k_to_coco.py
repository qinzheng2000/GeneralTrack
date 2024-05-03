import os
import numpy as np
import json
import cv2


# Use the same script for MOT16
DATA_PATH = 'datasets/detections_GHOST/bdd100k'
image_PATH = 'datasets/bdd100k/images/track'
OUT_PATH = os.path.join('datasets/bdd100k/images/track', 'annotations')

# SPLITS = ['train', 'val', 'test']  # --> split training data to train_half and val_half.
SPLITS = ['val']
HALF_VIDEO = True
CREATE_SPLITTED_ANN = True
CREATE_SPLITTED_DET = True

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


if __name__ == '__main__':

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:
        data_path = os.path.join(DATA_PATH, split)
        out_path = os.path.join(OUT_PATH, '{}.json'.format(split))
        out = {'images': [], 'annotations': [], 'videos': [],
               'categories': [{'id': 1, 'name': 'pedestrian'}]}
        seqs = os.listdir(data_path)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        tid_curr = 0
        tid_last = -1
        for seq in sorted(seqs):
            video_cnt += 1  # video sequence number.
            # if video_cnt ==2: break
            out['videos'].append({'id': video_cnt, 'file_name': seq})
            seq_path = os.path.join(image_PATH, split,seq)
            images = sorted(os.listdir(seq_path))
            num_images = len([image for image in images if 'jpg' in image])

            # detection文件
            det_path  = os.path.join('datasets/detections_GHOST/bdd100k', split, seq, 'det/yolox_dets.txt')
            dets = np.loadtxt(det_path, dtype=np.float32, delimiter=',')
            frame_index = dets[:, 0]
            dets = dets[:, 2:8]
            dets[:, 2:4] += dets[:, :2]




            for i in range(num_images):
                img = cv2.imread(os.path.join(image_PATH, split,'{}/{}'.format(seq, images[i])))
                height, width = img.shape[:2]
                # detetcion结果存进json文件里，output的格式：（n，6）（x1y1x2y2，score，cls）
                index = frame_index == (i + 1)
                det = dets[index]

                image_info = {'file_name': '{}/{}'.format(seq, images[i]),  # image name.
                              'id': image_cnt + i + 1,  # image number in the entire training set.
                              'frame_id': i + 1 ,  # image number in the video sequence, starting from 1.
                              'prev_image_id': image_cnt + i if i > 0 else -1,  # image number in the entire training set.
                              'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                              'video_id': video_cnt,
                              'height': height, 'width': width,
                              'detection': det
                              }

                out['images'].append(image_info)





            print('{}: {} images'.format(seq, num_images))
            image_cnt += num_images
            print(tid_curr, tid_last)




        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'), cls=NpEncoder)