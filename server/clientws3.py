# import necessary libs
import  cv2
from vidgear.gears import CamGear, WriteGear
from vidgear.gears.helper import create_blank_frame, reducer, retrieve_best_interpolation
from utils.solutions import object_counter
from utils.general import image_resize, getConfProperty, setProperty
from urllib.parse import urlparse
import cv2
import torch
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.utils import LOGGER, ROOT, is_colab, is_kaggle, ops
from server.patched_yolo_infer.functions_extra import Segment_Stream_Class
import traceback
import threading
import asyncio
import time
import numpy as np
import os
# create your own custom streaming class
class Custom_Stream_Class:

    def __init__(self, stride=4):        
        self.running = True
        self.source, self.thrP, self.model, self.modelName, self.type, self.stride, self.sourceVideo =None, None, None, None, None, None, None
        self.seg, self.start = None, None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
      
        # self.model = YOLO("yolov8n-seg.pt").to(self.device)
        # self.model = FastSAM('FastSAM-x.pt')
        # self.modelSeg = modelSeg
        self.cv2= cv2 
        self.SourceType="yt"
        # self.setType(type)
        # self.setModel(model)
        self.setStride(stride)
        # self.seg = Segment_Stream_Class (self.model)
        self.countImg=0 
        # self.stride=stride
        # self.wait=self.getWaitFrame()
        output_params = {"-f": "rtsp", "-rtsp_transport": "tcp", "-bufsize":"100k"}   

        rtspServer, _ =getConfProperty("rtspServer")  
        
        self.writer = WriteGear(output="rtsp://%s:8554/mystream" % rtspServer, logging=True, **output_params)
        # output_params = {
        #     "-clones": ["-f", "lavfi", "-i", "anullsrc"],
        #     "-vcodec": "libx264",
        #     "-preset": "medium",
        #     "-b:v": "4500k",
        #     "-bufsize": "512k",
        #     "-pix_fmt": "yuv420p",
        #     "-f": "flv",
        # }



        # YOUTUBE_STREAM_KEY = "phbq-55ve-tah4-h6jk-a29q"

        # self.writer = WriteGear(
        #     output="rtmp://a.rtmp.youtube.com/live2/{}".format(YOUTUBE_STREAM_KEY),
        #     logging=False,
        #     **output_params
        # )
        self.thrP = threading.Thread(target=self.between_callback,  args=(), kwargs={})
        self.thrP.start()
        
        # self.default_img = cv2.imread('logo512.png', 0) 
    def setStride(self, stride):
        self.stride=stride
        self.start = time.time()
        self.timestamp=0
        # self.wait= self.getWaitFrame()

 
    def setModelAndType(self, model, modelName, type):
        self.model=model
        self.modelName=modelName
        if type=="segmentation":
            self.seg = Segment_Stream_Class (self.model)
        self.type=type


    # def getWaitFrame(self):
    #     _start = time.time()
    #     _timestamp = 0
    #     VIDEO_CLOCK_RATE = 90000
    #     fps=30/ self.stride
    #     VIDEO_PTIME = 1 / fps  # 30fps
    #     _timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
    #     return _start + (_timestamp / VIDEO_CLOCK_RATE) - time.time()


    def setVideo(self, sourceVideo=None):

        self.sourceVideo=sourceVideo
        self.running = True
        region_points, _ =getConfProperty("region_points")  
        
        # self.wait=self.getWaitFrame()

        # self.type=type
        if urlparse(self.sourceVideo).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):
            self.SourceType="yt"      
        else:
            self.SourceType="rtsp"        

        options = {"STREAM_RESOLUTION": "360p", "CAP_PROP_FRAME_WIDTH":640, "CAP_PROP_FRAME_HEIGHT":360 }
        self.source = CamGear(source=self.sourceVideo,  stream_mode=True if self.SourceType=="yt" else False,  logging=False, **options if self.SourceType=="yt" else {}).start()    
   

        self.counter=[]
        
        region_points_dict = [x for x in region_points if x['source'] == self.sourceVideo and x['available'] == 1][0]
        print("pasa 13")
        for i, rp in enumerate(region_points_dict["region_points"]):
            print("pasa i")
            ctr= object_counter.ObjectCounter()
            ctr.set_args(view_img=False,
                        reg_pts=rp,
                        classes_names=self.model.names,
                        draw_tracks=True,
                        reg_counts=region_points_dict["reg_counts"][i]
                        )
            print(i)
            self.counter.append(ctr)
        
        self.countImg=0    
        self.start = time.time()
        self.timestamp=0
        
        

    def read(self):
       
        if self.running:
            # read frame from provided source
            while True:
                if self.source is None:
                    break
                self.countImg= self.countImg+1
                # print(self.countImg)
                frame =  self.source.read()     
                
                if self.countImg % self.stride == 0:   
                    # check if frame is available
                    if frame is not None:
                        # path = os.path.dirname(os.path.realpath(__file__))
                        # cv2.imwrite(os.path.join(path , '4.jpg'), frame)
                        # cv2.imshow("RTSP View", frame)
                        # if self.SourceType=="rtsp":
                        #     frame = image_resize(frame, height = 360)

                        # if self.type=="detection":
                        #     dict_result=dict()
                        #     dict_result["verbose"] =False
                        #     try:
                        #         results = self.model.track(frame, persist=True,  imgsz=640,  show=False, **dict_result)                
                        #         for ctr in self.counter:
                        #             frame = ctr.start_counting(frame, results) 
                        #     except Exception as e:                                
                        #         traceback.print_exception(type(e), e, e.__traceback__)    
                        # else:
                        #     print(frame.size)
                        #     frame=self.seg.visualize_results_usual_yolo_inference(
                        #         frame,
                        #         # self.model,
                        #         # imgsz=720,
                        #         conf=0.35,
                        #         iou=0.7,
                        #         segment=True,
                        #         thickness=1,
                        #         show_boxes=False,
                        #         fill_mask=True,
                        #         alpha=0.8,
                        #         random_object_colors=True,
                        #         # list_of_class_colors=[(231, 64, 50),(231, 64, 50),(231, 64, 50),(231, 64, 50),(231, 64, 50),(231, 64, 50),(231, 64, 50),(231, 64, 50),(231, 64, 50)],
                        #         show_class=False,
                        #         dpi=150,
                        #     return_image_array=True,
                            
                        #         )
                        
                        return frame
                    else:
                        self.running = True
                        return None
      
        # return None-type
        return None
    
    def restart(self):  
        self.source.stop()
        print("cae en restart")

    def stop(self):
        self.running = False
        self.source.stop()
        self.writer.close()
        
        print("cae en stop")
        # close stream
        # if not self.source is None:
        #     self.source.stop()
        #     self.source.release()

    def between_callback(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        loop.run_until_complete(self.service())
        loop.close()

    async def service (self):
       
       
        VIDEO_CLOCK_RATE = 90000
        self.timestamp=0
        
        # cl=Custom_Stream_Class(model, modelSeg)
        blank_frame=np.zeros([360,640,3],dtype=np.uint8)
        black_frame=blank_frame[:]
        black_frame=create_blank_frame(frame=black_frame, text="")
        __frame_size_reduction = 20  # 20% reduction
                # retrieve interpolation for reduction
        __interpolation = retrieve_best_interpolation(
            ["INTER_LINEAR_EXACT", "INTER_LINEAR", "INTER_AREA"]
        )
        # youtube=False
        while True:
            try:    
                if self.running==False:
                    break       
                if self.source is None:
                    frame=black_frame  
                # else:
                #     youtube=True   
                # print(wait)
                fps=30/ self.stride
                VIDEO_PTIME = 1 / fps  # 30fps
                self.timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
                wait= self.start + (self.timestamp / VIDEO_CLOCK_RATE) - time.time()

                await asyncio.sleep(wait)
              
                frame=self.read()
                if frame is None:
                    frame=black_frame   

                f_stream=frame
                # f_stream =  reducer(
                #             frame,
                #             percentage=__frame_size_reduction,
                #             interpolation=__interpolation,
                #         ) 
                # if f_stream is not None and youtube:  
                if f_stream is not None:                      
                    self.writer.write(f_stream)

            except Exception as e:          
                self.running=False
                LOGGER.error("An exception occurred in service : %s" % e)     
                traceback.print_exception(type(e), e, e.__traceback__)
                break
            