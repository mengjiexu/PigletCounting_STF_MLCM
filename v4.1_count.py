import sys
sys.path.append('..')
import os
from tqdm import tqdm
import numpy as np
import cv2
import pandas as pd
import pynvml
import time

import pycuda.autoinit  # noqa F401
import pycuda.driver as cuda

from YOLOv8TRT import PigDetectYOLO8TRT
from sort.sort import Sort


#  使用YOLO8的v1版本，主要是增加了置信度loss的权重
# 在v1版本上，使用yolo8l的预训练模型微调

class Pig():
    def __init__(self, frameidx, start_box, pigid) -> None:
        self.pigid = pigid
        self.boxes = [start_box]
        x1, y1, x2, y2 = start_box
        self.end_x = self.start_x = (x1+x2)/2
        self.end_y = self.start_y = (y1+y2)/2
        self.frameidxs = [frameidx]
    
    def update(self, box, frameidx):
        x1, y1, x2, y2 = box
        self.boxes.append(box)
        self.end_x = (x1+x2)/2
        self.end_y = (y1+y2)/2
        self.frameidxs.append(frameidx)


def count_pig(pig_dict, cut_x):
    """
    划线统计方法
    """
    count = 0
    for pigid in pig_dict:
        pig = pig_dict[pigid]
        if (pig.start_x - cut_x)*(pig.end_x - cut_x)<0:
            if pig.start_x>pig.end_x:
                count += 1
            else:
                count -= 1
    return count


def get_best_gpu(): # return gpuid with largest free memory.
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    deviceMemory = []
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        deviceMemory.append(mem_info.free)
    deviceMemory = np.array(deviceMemory, dtype=np.int64)
    best_device_index = np.argmax(deviceMemory)
    return best_device_index


def process_one_video(args):
    time.sleep(5)
    videof, output_videof, model_file, output_csvf, real_count = args
    video_output_csvf = output_videof.replace('.mp4', '.csv')
    output_record_csvf = video_output_csvf.replace('_output.csv', '_record.csv')
    # 选择未使用的gpu计算
    useid = int(get_best_gpu())
    
    # print('useid', useid)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(useid)
    ctx = cuda.Device(useid).make_context()
    model = PigDetectYOLO8TRT(model_file)

    
    video = cv2.VideoCapture(videof)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = cv2.VideoWriter(output_videof, \
                                   cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))
    df_record = pd.DataFrame(columns=['frameidx', 'x1', 'y1', 'x2', 'y2', 'label', 'score'])
    df_output = pd.DataFrame(columns=['frameidx', 'x1', 'y1', 'x2', 'y2', 'label', 'score', 'pigid', 'count'])
    # tracker
    tracker = Sort(max_age=1, min_hits=3)
    print('write to', output_videof, fps, [frame_w, frame_h])
    bar = tqdm(total=video.get(7))
    pig_dict = {}  # {'id' : Pig()}
    frameidx = 0
    pig_count = 0
    while True:
        bar.update()
        ret, frame = video.read()
        if frame is None:
            break
        boxes     = model.predict(frame)
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2, cls, score = list(map(float, box))
            df_record.loc[len(df_record)] = [frameidx, x1, y1, x2, y2, cls, score]
        classes   = boxes[:, -2]
        boxes     = boxes[:, :-2]
        pig_boxes = boxes[classes==0]
        track_boxes = tracker.update(pig_boxes)
        for box in track_boxes:
            x1, y1, x2, y2, pigid = list(map(int, box))
            if pigid not in pig_dict:
                pig_dict[pigid] = Pig(frameidx, [x1, y1, x2, y2], pigid)
            else:
                pig_dict[pigid].update([x1, y1,x2, y2], frameidx)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 128), 2)
            cv2.putText(frame, str(pigid), ((x1+x2)//2, (y1+y2)//2), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
            df_output.loc[len(df_output)] = [frameidx, x1, y1, x2, y2, 0, 1, pigid, pig_count]
        pig_count = count_pig(pig_dict, frame_w/2)
        cv2.line(frame, (frame_w//2, 0), (frame_w//2, frame_h), (255, 255, 0), 2)
        cv2.putText(frame, f'count:{pig_count}', (30, 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 255), 2)
        output_video.write(frame)
        frameidx += 1
    output_video.release()
    video.release()
    df_output.to_csv(video_output_csvf, index=False)
    df_record.to_csv(output_record_csvf, index=False)
    print('pred count:', pig_count)
    ctx.pop()
    return pig_count
    

if __name__ == '__main__':
    import sys
    
    videof = sys.argv[1]
    output_videof = sys.argv[2]
    # model_file = sys.argv[3]
    model_file = './models/voc_pig_v4.1/train/weights/best.engine'
    output_csvf = sys.argv[4]
    real_count = sys.argv[5]
    
    process_one_video([videof, output_videof, model_file, output_csvf, real_count])
    
    