import argparse
from importlib.resources import path
import logging
import time
import csv

import cv2
import numpy as np
import tensorflow as tf
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import psutil

# If pafprocess error, build it from source as follows
# $ cd core/tf_pose/pafprocess/
# $ swig -python -c++ pafprocess.i 
# $ python setup.py build_ext --inplace

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='bigwind.mov')
    parser.add_argument('--resolution', type=str, default='464x400', help='network input resolution. default=464x400')
    parser.add_argument('--model', type=str, default='mobilenet_v2_large', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    # logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    cap = cv2.VideoCapture(args.video)
    # cap.set(cv2.CAP_PROP_FPS, int(30))

    if cap.isOpened() is False:
        print("Error opening video stream or file")
    video_landmarks = []
    fps_list = []
    while cap.isOpened():
        ret_val, image = cap.read()
        if not ret_val:
            break
        humans = e.inference(image, resize_to_default = True, upsample_size = 4.0)
        if len(humans) > 0:
            bodyParts = humans[0].body_parts
            frame_landmarks = []
            for k, v in bodyParts.items():
                x, y = v.x, v.y
                frame_landmarks.append(x)
                frame_landmarks.append(y)
            if not args.showBG:
                image = np.zeros(image.shape)
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

            video_landmarks.append(frame_landmarks)

            with open("landmarks.csv", "w") as file:
                csvwriter = csv.writer(file)
                csvwriter.writerows(video_landmarks)

            # print(video_landmarks)
        t2 = time.time()
        fps = int(1 / (t2 - fps_time))
        # print(t2-fps)
        fps_list.append(fps)

        if len(fps_list) > 20:
            fps_mean = int(sum(fps_list)/len(fps_list))

            cv2.putText(image, "FPS: " f"{fps_mean}", (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
        cv2.resize(image, (1920, 1080))
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
# logger.debug('finished+')
# For single process
# highProcess = psutil.pids()[-1]
# p = psutil.Process(highProcess)
# print(p.name)
# p.terminate()