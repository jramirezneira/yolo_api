# import necessary libs
import  cv2
from vidgear.gears import  CamGear
from utils.solutions import object_counter
from utils.general import image_resize, getConfProperty, setProperty
from urllib.parse import urlparse
import cv2
import torch
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.utils import LOGGER, ROOT, is_colab, is_kaggle, ops
from patched_yolo_infer.functions_extra import Segment_Stream_Class
import traceback

# create your own custom streaming class
class Custom_Stream_Class:

    def __init__(self, model, modelSeg):        
        self.running = True
        self.source=None
     
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model =model
        # self.model = YOLO("yolov8n-seg.pt").to(self.device)
        # self.model = FastSAM('FastSAM-x.pt')
        self.modelSeg = modelSeg
        self.cv2= cv2 
        self.SourceType="yt"
        self.type="detection"
        self.seg = Segment_Stream_Class (self.modelSeg)
        # self.default_img = cv2.imread('logo512.png', 0) 

    def change(self, source=None, type="detection"):
        self.type=type
        if urlparse(source).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):
            self.SourceType="yt"      
        else:
            self.SourceType="rtsp"        

        options = {"STREAM_RESOLUTION": "720p", "CAP_PROP_FRAME_WIDTH":1280, "CAP_PROP_FRAME_HEIGHT":720 }
        self.source = CamGear(source=source,  stream_mode=True if self.SourceType=="yt" else False,  logging=False, **options if self.SourceType=="yt" else {}).start()    
   

        self.counter=[]
        region_points, self.stride =getConfProperty("region_points", "stride")
        region_points_dict = [x for x in region_points if x['source'] == source and x['available'] == 1][0]
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
        self.running = True
        self.countImg=0    
        

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
                        # cv2.imshow("RTSP View", frame)
                        if self.SourceType=="rtsp":
                            frame = image_resize(frame, height = 720)

                        if self.type=="detection":
                            dict_result=dict()
                            dict_result["verbose"] =False
                            try:
                                results = self.model.track(frame, persist=True,  imgsz=640,  show=False, **dict_result)                
                                for ctr in self.counter:
                                    frame = ctr.start_counting(frame, results) 
                            except Exception as e:                                
                                traceback.print_exception(type(e), e, e.__traceback__)

                                           
                            
                        else:
                            frame=self.seg.visualize_results_usual_yolo_inference(
                                frame,
                                # self.model,
                                # imgsz=720,
                                conf=0.35,
                                iou=0.7,
                                segment=True,
                                thickness=4,
                                show_boxes=False,
                                fill_mask=True,
                                alpha=0.8,
                                random_object_colors=False,
                                # list_of_class_colors=[(231, 64, 50),(231, 64, 50),(231, 64, 50),(231, 64, 50),(231, 64, 50),(231, 64, 50),(231, 64, 50),(231, 64, 50),(231, 64, 50)],
                                show_class=True,
                                dpi=150,
                            return_image_array=True,
                            
                                )
                        
                        return frame
                    else:
                        # print("self.source4")
                        # # signal we're not running now
                        # print("cae en stop1")
                        self.running = True
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