import os
import sys
import json
import cv2
import glob as gb
import numpy as np


def colormap(rgb=False):
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list

def paint(bboxes, img_p, save_p):
    color_list = colormap()

    img = cv2.imread(img_p)
    for bbox in bboxes:
        bbox_ = [float(bbox[0]), float(bbox[1]),
                float(bbox[0]) + float(bbox[2]),
                float(bbox[1]) + float(bbox[3]), bbox[4]]
        cv2.rectangle(img, (int(bbox_[0]), int(bbox_[1])), (int(bbox_[2]), int(bbox_[3])),
                      color_list[int(bbox_[4]+1) % 79].tolist(), thickness=2)
        cv2.putText(img, "{}".format(bbox_[4]+1), (int(bbox_[0]), int(bbox_[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    color_list[int(bbox_[4]+1) % 79].tolist(), 2)
    cv2.imwrite(save_p, img)


def txt2imge(visual_path, img_path, txt_path):
    print("Starting txt2img")
    color_list = colormap()

    if not os.path.exists(visual_path):
        os.makedirs(visual_path)

    video_names = os.listdir(img_path)
    for video_name in video_names:
        if video_name!= 'b2036451-aa924fd1':continue
        imgs_name = sorted(os.listdir(os.path.join(img_path, video_name)))
        # txt = np.loadtxt(os.path.join(txt_path, video_name+'.txt'), dtype=np.float32, delimiter=',').astype(float)
        if not os.path.exists(os.path.join(visual_path, video_name)):
            os.makedirs(os.path.join(visual_path, video_name))

        txt_dict = dict()
        # with open(os.path.join(txt_path, video_name+'.txt'), 'r') as f:
        # with open(os.path.join(txt_path, video_name,  'det/yolox_dets.txt'), 'r') as f:
        with open(os.path.join(txt_path, video_name +'.json'), 'r', encoding='utf-8') as f:
            datas = json.load(f)

        for data in datas:
            img = cv2.imread(os.path.join(img_path,video_name,data['name']))
            # img = cv2.imread(os.path.join(img_path, data['name']))

            for box in data['labels']:
                bbox = [box['box2d']['x1'], box['box2d']['y1'], box['box2d']['x2'], box['box2d']['y2'], int(box['id']),
                        box['category']]
                if (bbox[4] in [41, 45] ):
                # if bbox[4] in [11875]:
                    if bbox[4] == 40: color = color_list[int(bbox[4] ) % 79].tolist()
                    if bbox[4] == 41: color = (0,255,255)
                    if bbox[4] == 45: color =(0,128,255)
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                  color, thickness=13)
                    # cv2.putText(img, "{}".format(bbox[5]), (int((int(bbox[0])+int(bbox[2]))/2), int((int(bbox[1])+int(bbox[3]))/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    #             color_list[bbox[4] % 79].tolist(), 1)
                    cv2.putText(img, "{}".format(bbox[4]), (int(bbox[0]) + 100, int(bbox[1])-15), cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, color, 3)
                    cv2.putText(img, "{}".format(bbox[5]), (int(bbox[0]), int(bbox[1])-15), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                color, 3)
                elif (bbox[4] in [40] ) & ('139' in data['name'].split('-')[2]):
                # if bbox[4] in [11875]:
                    color = color_list[int(bbox[4] ) % 79].tolist()
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                  color, thickness=13)
                    # cv2.putText(img, "{}".format(bbox[5]), (int((int(bbox[0])+int(bbox[2]))/2), int((int(bbox[1])+int(bbox[3]))/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    #             color_list[bbox[4] % 79].tolist(), 1)
                    cv2.putText(img, "{}".format(bbox[4]), (int(bbox[0]) + 100, int(bbox[1])-15), cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, color, 3)
                    cv2.putText(img, "{}".format(bbox[5]), (int(bbox[0]), int(bbox[1])-15), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                color, 3)
                else:
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                  color_list[int(bbox[4] )% 79].tolist(), thickness=1)
                    # cv2.putText(img, "{}".format(bbox[5]), (int((int(bbox[0])+int(bbox[2]))/2), int((int(bbox[1])+int(bbox[3]))/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    #             color_list[bbox[4] % 79].tolist(), 1)
                    cv2.putText(img, "{}".format(bbox[4]), (int(bbox[0])+50, int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                color_list[int(bbox[4] ) % 79].tolist(), 2)
                    cv2.putText(img, "{}".format(bbox[5]), (int(bbox[0]) , int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                color_list[int(bbox[4] ) % 79].tolist(), 2)
            cv2.imwrite(os.path.join(visual_path, data['name']), img)
            print(os.path.join(visual_path, data['name']))
        print('img', "Done")
print("txt2img Done")


def img2video(visual_path, gt_or_mymodel):
    print("Starting img2video")

    img_paths = gb.glob(visual_path + gt_or_mymodel + "/*.jpg")
    fps = 16
    size = (1920, 1080)
    videowriter = cv2.VideoWriter(visual_path + gt_or_mymodel + "_video.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

    for img_path in sorted(img_paths):
        img = cv2.imread(img_path)
        img = cv2.resize(img, size)
        videowriter.write(img)

    videowriter.release()
    print("img2video Done")


if __name__ == '__main__':
    img_path = ''
    txt_path = ''

    visual_path = ''
    txt2imge(visual_path, img_path, txt_path)
