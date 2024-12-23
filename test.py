import cv2
import yolov8_runner as rknn_yolov8_demo
import time

def running_test(runner,img,times = 1000):
    start_time = time.time()
    for i in range(times):
        result = runner.inference(img)
    print('推理耗时MS:', (time.time() - start_time) * 1000/times)
if __name__ == '__main__':
    run1 = rknn_yolov8_demo.yolov8_runner('./model/yolov8.rknn',0)
    img = cv2.imread('./model/bus.jpg')

    running_test(run1,img,1000)