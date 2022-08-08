import argparse
import logging
import time

import cv2
import numpy as np
import psutil
import tensorflow as tf
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import multiprocessing

from numba import cuda
tf.keras.backend.clear_session()
import json

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def main():
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    fps_time = 0

    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    counter = 1
    while True:
        ret_val, image = cam.read()

        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        frame_landmarks = {"person_a":{}, "person_b" : {}, "person_c" : {}, "person_d" : {}}
        if len(humans) == 0:
            continue

        elif len(humans) == 1:
            first_person = humans[0].body_parts
            
            frame = {}
            frame_list = []
            for _, v_a in first_person.items():
                body_index_a = v_a.part_idx
                x_a, y_a = v_a.x, v_a.y
                frame[body_index_a] = [round(x_a, 3), round(y_a, 3)]
                frame_list.append(frame)
                frame_landmarks["person_a"] = frame_list
            # with open("frame_landmarks.json", "w") as f:
            #     json.dump(frame_landmarks, f)
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        elif len(humans) == 2:
            first_person = humans[0].body_parts
            second_person = humans[1].body_parts
            # third_person = humans[2].body_parts
            # fourth_person = humans[3].body_parts

            for _, v_a in first_person.items():
                body_index_a = v_a.part_idx
                x_a, y_a = v_a.x, v_a.y
                frame[body_index_a] = [round(x_a, 3), round(y_a, 3)]
                frame_list.append(frame)
                frame_landmarks["person_a"] = frame_list
            # with open("frame_landmarks.json", "w") as f:
            #     json.dump(frame_landmarks, f)
                
            for _, v_b in second_person.items():
                body_index_b = v_b.part_idx
                x_b, y_b = v_b.x, v_b.y
                frame[body_index_b] = [round(x_b, 3), round(y_b, 3)]
                frame_list.append(frame)
                frame_landmarks["person_b"] = frame_list
            # with open("frame_landmarks.json", "w") as f:
            #     json.dump(frame_landmarks, f)

            # for _, v_c in third_person.items():
            #     body_index_c = v_c.part_idx
            #     x_c, y_c = v_c.x, v_c.y

            # for _, v_d in fourth_person.items():q
            #     body_index_d = v_d.part_idx
            #     x_d, y_d = v_d.x, v_d.y 
            
            # frame_landmarks["person_c"][body_index_c] = [round(x_c, 3), round(y_c, 3)]
            # frame_landmarks["person_d"][body_index_d] = [round(x_d, 3), round(y_d, 3)]
            # with open("frame_landmarks.json", "w") as f:
            #     json.dump(frame_landmarks, f)
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        else:
            image = np.zeros(image.shape)
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        cv2.resize(image, (1080, 720), cv2.INTER_CUBIC)
        counter += 1
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.putText(image, f"Persons detected = {len(humans)}", (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1, (127, 255, 255), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # logger.debug('finished+')

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # p = multiprocessing.Process(target=main)
    # p.start()
    # p.join()
    main()



# For single process
highProcess = psutil.pids()[-1]
p = psutil.Process(highProcess)
print(p.name)
p.terminate()