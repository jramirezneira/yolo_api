# import necessary libs
import uvicorn, cv2
from vidgear.gears.asyncio import WebGear_RTC
from vidgear.gears import NetGear, CamGear
import os
from ultralytics import YOLO
from ultralytics.solutions import object_counter
from utils.general import image_resize, getConfProperty, setProperty
from urllib.parse import urlparse
import gc
import cv2
import numpy as np
import requests
import torch
from PIL import Image

from ultralytics.data.augment import LetterBox
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.utils import LOGGER, ROOT, is_colab, is_kaggle, ops
from ultralytics.utils.checks import check_requirements

# create your own custom streaming class
class Custom_Stream_Class:

    def __init__(self):

        
        self.running = True
        self.source=None
        self.names = {   0: 'person',  1: 'bicycle',  2: 'car',  3: 'motorcycle',  4: 'airplane',  5: 'bus',  6: 'train',  7: 'truck',
            8: 'boat',  9: 'traffic light',  10: 'fire hydrant',  11: 'stop sign',  12: 'parking meter',  13: 'bench',  14: 'bird',
            15: 'cat',  16: 'dog',  17: 'horse',  18: 'sheep',  19: 'cow',  20: 'elephant',  21: 'bear',  22: 'zebra',  23: 'giraffe',
            24: 'backpack',  25: 'umbrella',  26: 'handbag',  27: 'tie',  28: 'suitcase',  29: 'frisbee',  30: 'skis',  31: 'snwboard',
            32: 'sports ball',  33: 'kite',  34: 'baseball bat',  35: 'baseball glove',  36: 'skateboard',  37: 'surfboard',  38: 'tennis racket',
            39: 'bottle',  40: 'wine glass',  41: 'cup',  42: 'fork',  43: 'knife',  44: 'spoon',  45: 'bowl',  46: 'banana',  47: 'apple',
            48: 'sandwich',  49: 'orange',  50: 'broccoli',  51: 'carrot',  52: 'hot dog',  53: 'pizza',  54: 'donut',  55: 'cake',
            56: 'chair',  57: 'couch',  58: 'potted plant',  59: 'bed',  60: 'dining table',  61: 'toilet',  62: 'tv',  63: 'laptop',
            64: 'mouse',  65: 'remote',  66: 'keyboard',  67: 'cell phone',  68: 'microwave',  69: 'oven',  70: 'toaster',  71: 'sink',
            72: 'refrigerator',  73: 'book',  74: 'clock',  75: 'vase',  76: 'scissors',  77: 'teddy bear',  78: 'hair drier',  79: 'toothbrush'}


       
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO("yolov8n.pt").to(self.device)
        self.cv2= cv2 
        self.type="yt"
        self.default_img = cv2.imread('logo512.png', 0) 

    def change(self, source=None):
        if urlparse(source).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):
            self.type="yt"      
        else:
            self.type="rtsp"        

        options = {"STREAM_RESOLUTION": "720p", "CAP_PROP_FRAME_WIDTH":1280, "CAP_PROP_FRAME_HEIGHT":720 }
        self.source = CamGear(source=source,  stream_mode=True if self.type=="yt" else False,  logging=False, **options if self.type=="yt" else {}).start()    
        
        # options = {"CAP_PROP_FRAME_WIDTH":1280, "CAP_PROP_FRAME_HEIGHT":720 }
        # self.source = CamGear(source=source,  stream_mode=False,  logging=True).start()     

        self.counter=[]
        region_points, self.stride =getConfProperty("region_points", "stride")
        region_points_dict = [x for x in region_points if x['source'] == source and x['available'] == 1][0]
        print("pasa 13")
        for i, rp in enumerate(region_points_dict["region_points"]):
            print("pasa i")
            ctr= object_counter.ObjectCounter()
            ctr.set_args(view_img=False,
                        reg_pts=rp,
                        classes_names=self.names,
                        draw_tracks=True,
                        reg_counts=region_points_dict["reg_counts"][i]
                        )
            print(i)
            self.counter.append(ctr)
        self.running = True
        self.countImg=0
        

    def read(self):
        
       
        # if self.source is None:
        #     print("cae en stop2")
        #     return self.default_img
        # check if we're still running
       
        if self.running:
            # read frame from provided source
            while True:
                if self.source is None:
                    break
                self.countImg= self.countImg+1
                frame =  self.source.read()     
                
                if self.countImg % self.stride == 0:   
                    # check if frame is available
                    if frame is not None:
                        # cv2.imshow("RTSP View", frame)
                        if self.type=="rtsp":
                            frame = image_resize(frame, height = 720)
                        dict_result=dict()
                        dict_result["verbose"] =False
                        results = self.model.track(frame, persist=True, imgsz=640, show=False, **dict_result)                
                        for ctr in self.counter:
                            frame = ctr.start_counting(frame, results) 
                        return frame
                    else:
                        print("self.source4")
                        # signal we're not running now
                        print("cae en stop1")
                        self.running = False
                        return None
                            # return self.default_img
            # else:
            #     return None
      
        # return None-type
        return None

    def stop(self):
        self.running = False
        print("cae en stop2")
        # close stream
        # if not self.source is None:
        #     self.source.stop()
        #     self.source.release()