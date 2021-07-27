import cv2
import time
import os
import sys
import importlib
import time
import shutil
from tqdm import tqdm
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_PATH)
sys.path.append(BASE_PATH)
importlib.reload(sys)
from YoloV5Detector.V5Detector import Detector


def inference_single_image(weights_path, thresh, src_img, dst_img, cls, colors=None, gpu_id='0'):
    det = Detector(weights_path, gpu_id=gpu_id, colors=colors)
    for i in range(100):
        t1 = time.time()
        img = cv2.imread(src_img)
        img_res, det_res = det.detect(img, cls, thresh)
        t2 = (time.time() - t1) * 1000
        print("inference time:{} ms".format(t2))
        img_res = det.draw_box(img, det_res)
        det.print_result(det_res)
    cv2.imwrite(dst_img, img_res)

def get_images_from_dir(imgPath):
    imagelist = os.listdir(imgPath)
    image_dic = []
    for imgname in imagelist:
        if (imgname.endswith(".jpg")):
            imgp = imgPath + imgname
            image_dic.append(imgp)
    return image_dic

def inference_images(weights_path, thresh, src_dir, dst_dir, cls, colors=None, gpu_id='0'):
    det = Detector(weights_path, gpu_id=gpu_id, colors=colors)
    images_dic = get_images_from_dir(src_dir)
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.mkdir(dst_dir)
    for img_dic in images_dic:
        t1 = time.time()
        img = cv2.imread(img_dic)
        img_res, det_res = det.detect(img, cls, thresh)
        t2 = (time.time() - t1) * 1000
        print("{} inference time:{} ms".format(img_dic.split('/')[-1], t2))
        img_res = det.draw_box(img, det_res)
        det.print_result(det_res)
        dst_img = dst_dir + img_dic.split('/')[-1]
        cv2.imwrite(dst_img, img_res)

def inference_videos(weights_path, thresh, src_video, dst_video, cls, colors=None, gpu_id='0'):
    det = Detector(weights_path, gpu_id=gpu_id, colors=colors)
    cap = cv2.VideoCapture(src_video)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("fps: {}\nsize: {}\ntotal_frame:{}".format(fps, size, total_frame))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(dst_video, fourcc, fps, size)

    for i in tqdm(range(total_frame)):
        success, frame = cap.read()
        if not success:
            print("this video is over!")
            break
        img_res, det_res = det.detect(frame, cls, thresh)
        img_res = det.draw_box(img_res, det_res)
        out.write(img_res)


if __name__ == '__main__':
    weights_path = "weights/5.0/yolov5s.pt"
    thresh = 0.3
    src_img = "samples/images/zidane.jpg"
    dst_img = "samples/images/zidane_res.jpg"
    cls = ['person', 'bus', 'horse', 'dog']
    colors = {0: (0, 0, 255), 5: (0, 255, 0)}
    gpu_id = '0'
    src_dir = "samples/images/"
    dst_dir = "samples/images/res/"
    src_video = "samples/videos/person.mp4"
    dst_video = "samples/videos/person_res.mp4"
    # inference_single_image(weights_path, thresh, src_img, dst_img, cls, colors, gpu_id)
    inference_images(weights_path, thresh, src_dir, dst_dir, cls, colors, gpu_id)
    # inference_videos(weights_path, thresh, src_video, dst_video, cls, colors, gpu_id)




