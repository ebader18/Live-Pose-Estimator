# source = 'http://10.0.0.44:8080/video'
# f_x, f_y = 522.5802793786835, 522.7410914643813
# c_x, c_y = 353.65300765751493, 224.41622041388987

import cv2
import numpy as np
from pose_estimator import pose_estimator
import time
import threading
import argparse

latest_image = None
lock = threading.Lock()
exit_flag = False


def capture_image(source):
    global latest_image, lock, exit_flag
    cap = cv2.VideoCapture()
    cap.open(source)
    while not exit_flag:
        ret, frame = cap.read()
        if ret:
            with lock:
                latest_image = frame
    cap.release()


def transform_to_global(R, t, point):
    return R @ point + t


if __name__ == '__main__':
    # Set up argparse
    parser = argparse.ArgumentParser(description="Captures live images from phone and compute pose.")
    parser.add_argument("ip", help="IP address for ip-camera or phone.")
    parser.add_argument("intrinsic", nargs=4, type=float, help="Intrinsic parameters for camera in following order 'fx fy cx cy', e.g. '522.5 522.7 353.6 224.4'")
    args = parser.parse_args()

    source = 'http://' + f'{args.ip}' + ':8080/video'
    f_x = args.intrinsic[0]
    f_y = args.intrinsic[1]
    c_x = args.intrinsic[2]
    c_y = args.intrinsic[3]
    K = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ])

    capture_thread = threading.Thread(target=capture_image, args=(source,))
    capture_thread.start()

    pose = pose_estimator(K, max_feature=10000)
    global_R = np.eye(3)
    global_t = np.zeros((3, 1))

    info_img = np.ones((480, 640*2, 3), dtype=np.uint8) * 255
    try:
        while True:
            t0 = time.time()
            with lock:
                if latest_image is not None:
                    img = latest_image.copy()
                    ret, R, t, match_img = pose.compute_pose(img)
                    if ret:
                        # print(f'Translation: {t[0]}, {t[1]}, {t[2]}')
                        global_t = global_t + global_R @ t
                        global_R = R @ global_R
                        h, w = info_img.shape[:2]
                        center = np.array([w / 2, h / 2, 1]).reshape(-1, 1)
                        global_center = transform_to_global(global_R, global_t, center)
                        x, y = int(global_center[0, 0]), int(global_center[1, 0])
                        cv2.circle(info_img, (640+x//5, 240+y//5), 2, (0, 0, 255), -1)
                        img = np.concatenate((match_img, info_img), axis=0)
                        cv2.imshow('Latest Image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            t1 = time.time()
            print(f'Duration: {1000.*(t1-t0):.2f}ms')
    finally:
        exit_flag = True
        capture_thread.join()
        cv2.destroyAllWindows()
