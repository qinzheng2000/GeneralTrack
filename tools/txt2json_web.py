import os
import numpy as np
import json
import cv2


DATA_PATH = 'YOLOX_outputs/yolox_x/BDD100K/track_resultstest'
vedio_PATH = 'datasets/bdd100k/images/track/test'
OUT_PATH = DATA_PATH + '_json'

BDD_NAME_MAPPING = {
    1: "pedestrian", 2: "rider", 3: "car", 4: "truck",
    5: "bus", 6: "train", 7: "motorcycle", 8: "bicycle"}

if __name__ == '__main__':
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    seqs = os.listdir(DATA_PATH)
    for seq in sorted(seqs):
        txt_path = os.path.join(DATA_PATH, seq)
        dets = np.loadtxt(txt_path, dtype=np.float32, delimiter=',').astype(float)

        # 得到帧数，循环每一帧
        video_name = seq.split('.')[0]
        img_path = os.path.join(vedio_PATH, video_name)
        imgs_name = sorted(os.listdir(img_path))

        frame_num = len(imgs_name)

        # 在这里处理所有的track_id
        track_id = dets[:,1]
        t_id = list(set(list(track_id.tolist())))
        for id in t_id:
            # 这个id所有的ann
            a = track_id == id
            ann = dets[a]
            c = ann[:, 7]
            c = np.array(list(c.tolist()), dtype=int)
            count = np.bincount(c)
            c_ = np.argmax(count)
            if c.max() != c_ or c.min()!= c_:
                print('s')
            dets[a, 7] = c_

        out = []
        for i in range(frame_num):
            index = dets[:,0] == (i + 1)
            det = dets[index]
            labels = []
            for j in range(det.shape[0]):
                box2d_tlwh = det[j][2:6]
                box2d_tlwh[2:4] += box2d_tlwh[:2]

                box2d = {'x1': box2d_tlwh[0],  # video_name
                              'y1': box2d_tlwh[1],  # imgs_name
                              'x2': box2d_tlwh[2],  # image number in the video sequence, starting from 0.
                              'y2': box2d_tlwh[3]
                              }

                a = int(det[j][7])
                object_info = {'category': BDD_NAME_MAPPING[1+int(det[j][7])],  # video_name
                              'id': int(det[j][1]),  # imgs_name
                              'box2d': box2d,  # image number in the video sequence, starting from 0.
                              'score': det[j][6]
                              }
                labels.append(object_info)

            image_info = {
                         'name': video_name + '/' + imgs_name[i],  # imgs_name
                          'labels': labels
                          }



            out.append(image_info)
        b = os.path.join(OUT_PATH, video_name+'.json')
        json.dump(out, open(os.path.join(OUT_PATH, video_name+'.json'), 'w'))