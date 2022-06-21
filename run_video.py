import argparse
import logging
import time

import cv2
import numpy as np
import tensorflow as tf
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import psutil



# # logger = logging.getLogger('TfPoseEstimator-Video')
# logger.setLevel(logging.DEBUG)
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
# ch.setFormatter(formatter)
# logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='sample2.mp4')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    # logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_FPS, int(30))

    if cap.isOpened() is False:
        print("Error opening video stream or file")
    video_landmarks = []
    while cap.isOpened():
        ret_val, image = cap.read()

        humans = e.inference(image)
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
        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.resize(image, (1920, 1080))
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
# logger.debug('finished+')
# For single process
highProcess = psutil.pids()[-1]
p = psutil.Process(highProcess)
print(p.name)
p.terminate()