# import necessary libs
import  cv2
from vidgear.gears import CamGear, WriteGear
from vidgear.gears.helper import create_blank_frame, reducer, retrieve_best_interpolation
from utils.solutions import object_counter
from utils.solutions.object_counter import  extract_and_process_face_detaction
from ultralytics.engine.results import Results 
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
import queue
import os
import uvicorn
from vidgear.gears.asyncio import WebGear_RTC
from ultralytics import SAM
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image
import requests
from io import BytesIO
from matplotlib import pyplot



# create your own custom streaming class
class Clientws:
    def __init__(self):
        self.thrP = threading.Thread(target=self.setStartServer,  args=(), kwargs={})
        self.thrP.start()

    def setStartServer(self):
        options = {
        "custom_stream": Custom_Stream_Class(),
            "frame_size_reduction": 5,
            "enable_live_broadcast": True,
            "enable_infinite_frames":True,
            "CAP_PROP_FPS":20
            #   "CAP_PROP_FRAME_WIDTH":320, "CAP_PROP_FRAME_HEIGHT":240, "CAP_PROP_FPS":60
        }
        # options = {"custom_stream": Custom_Stream_Class()}

        # initialize WebGear_RTC app without any source
        # web = WebGear_RTC(source="rtsp://127.0.0.1:8554/mystream", logging=True)#, **options)
        web = WebGear_RTC( logging=True, **options)
        uvicorn.run(web(), host="0.0.0.0", port=5003)
            


class Custom_Stream_Class:

    def __init__(self, stride=4, **options):  
        self.source, self.thrP, self.model, self.modelName, self.type, self.stride, self.sourceVideo =None, None, None, None, None, None, None
        self.seg, self.start = None, None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cv2= cv2 
        self.SourceType="yt"
        self.setStride(stride)
        self.countImg=0 
        self.wait =0
        blank_frame=np.zeros([360,640,3],dtype=np.uint8)
        black_frame=blank_frame[:]
        black_frame=create_blank_frame(frame=black_frame, text="")
        __frame_size_reduction = 80  # 20% reduction
                # retrieve interpolation for reduction
        __interpolation = retrieve_best_interpolation(
            ["INTER_LINEAR_EXACT", "INTER_LINEAR", "INTER_AREA"]
        )
        self.black_frame =  reducer(
                            black_frame,
                            percentage=__frame_size_reduction,
                            interpolation=__interpolation,
                        )  
        self.lastImage= self.black_frame
        self.fps=30
        self.size_prev=92
        output_params = {"-f": "rtsp", "-rtsp_transport": "tcp", "-bufsize":"100k"}   

        rtspServer, _ =getConfProperty("rtspServer")  
        
        # self.writer = WriteGear(output="rtsp://%s:8554/mystream" % rtspServer, logging=True, **output_params)
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
        self.__queue = None
        self.__queue = queue.Queue(maxsize=96)
        self.thrP = threading.Thread(target=self.between_callback,  args=(), kwargs={})
        self.thrP.start()
        self.dict_result=dict()
        self.dict_result["verbose"] =False

        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print(f'Using device: {self.device}')
        # self.mtcnn = MTCNN(image_size=160, keep_all=True, post_process=True, device=self.device)
        self.mtcnn =MTCNN(image_size=160, keep_all=True, post_process=True)
        
        self.resnet = None
        
        # self.resnet = self.resnet.to( device=self.device)

        # workers = 0 if os.name == 'nt' else 4

        # dataset = datasets.ImageFolder('database')
        # dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
        # loader = DataLoader(dataset, collate_fn=self.collate_fn, num_workers=workers)



        self.candidate_image_embs=[]
        self.candidate_image_face=[]
        self.candidate_image_name=[]

        # for x, y in loader:
        #     candidate_emb, candidate_face = self.get_embedding_and_face(x)
        #     if candidate_emb is None:
        #         continue
        #     else:
        #         self.candidate_image_embs.append(candidate_emb)
        #         self.candidate_image_face.append(dataset.idx_to_class[y])
            
        # for candidate_image_path in candidate_image_paths:
        #     candidate_emb, candidate_face = self.get_embedding_and_face(candidate_image_path)
        #     if candidate_emb is None:
        #         continue
        #     else:
        #         self.candidate_image_embs.append(candidate_emb)
        #         self.candidate_image_face.append(candidate_face)
    #     self.mtcnn = MTCNN(
    #                 image_size=160, keep_all=True,
    #                 device=self.device)
        
    #     self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)


    #     workers = 0 if os.name == 'nt' else 4

        

    #     dataset = datasets.ImageFolder('database')
    #     dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    #     loader = DataLoader(dataset, collate_fn=self.collate_fn, num_workers=workers)
      
    #     aligned = []
    #     names = []
    #     for x, y in loader:
    #         # print(type(x))
    #         # faces, probs = self.mtcnn(x, return_prob=True)
    #         # if faces is None or len(faces) == 0:
    #         #     continue
    #         x_aligned, prob = self.mtcnn(x, return_prob=True)
    #         if x_aligned is not None:
    #             # print('Face detected with probability: {:8f}'.format(prob))
    #             aligned.append(x_aligned)
    #             names.append(dataset.idx_to_class[y])
    #     aligned = torch.stack(aligned).to(self.device)
    #     self.embeddings = self.resnet(aligned).detach().cpu()
    #     # print(embeddings)
    #         # embedding = self.resnet(faces[0].unsqueeze(0))
    #         # self.target_images.append(faces[0], embedding, dataset.idx_to_class(y))

    def collate_fn(self, x):
        return x[0]
        
    def get_embedding_and_face(self, image):
        """Load an image, detect the face, and return the embedding and face."""
        # results = self.model.track(image, persist=True, device=0,  imgsz=[384,640],  show=False, **self.dict_result)
        
        
       
        faces, prob = self.mtcnn(image, return_prob=True)
       

        if faces is None or len(faces) == 0:
            return None, None
        
        faces=faces.to(self.device)

        embedding = self.resnet(faces[0].unsqueeze(0))
        return embedding, faces[0]





    def setStride(self, stride):
        self.stride=stride
        self.start = time.time()
        self.timestamp=0
        self.fps=30/stride
        
        # self.wait= self.getWaitFrame()

 
    def setModelAndType(self, model, modelName, type):
        self.model=model
        self.modelName=modelName
        if "seg" in type:
            self.seg = Segment_Stream_Class (self.model)
        if "faceDetection" in type:
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
            self.resnet = self.resnet.to( device=self.device)

            dataset = datasets.ImageFolder('database')
            dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
            loader = DataLoader(dataset, collate_fn=self.collate_fn, num_workers=0)



            self.candidate_image_embs=[]
            self.candidate_image_face=[]
            self.candidate_image_name=[]

            for x, y in loader:
                candidate_emb, candidate_face = self.get_embedding_and_face(x)
                if candidate_emb is None:
                    continue
                else:
                    self.candidate_image_embs.append(candidate_emb)
                    self.candidate_image_face.append(dataset.idx_to_class[y])
     
        
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
        if "detection" in self.type:
        
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
        if "faceDetection" in self.type:
            self.object_counter= object_counter.ObjectCounter()
           
        
        

    def read(self):
        while True:
            if self.source is None or not self.countImg < self.source.frames:
                break
            self.countImg= self.countImg+1
            # print(self.countImg)
            frame =  self.source.read()     
            # path = os.path.dirname(os.path.realpath(__file__))
            # cv2.imwrite(os.path.join(path , 'Town of Collingwood.jpg'), frame)
            
            if self.countImg % self.stride == 0:   
                # check if frame is available
                if frame is not None:                    
                    # cv2.imshow("RTSP View", frame)
                    if self.SourceType=="rtsp":
                        frame = image_resize(frame, height = 360)
                        # path = os.path.dirname(os.path.realpath(__file__))
                        # cv2.imwrite(os.path.join(path , 'camara1.jpg'), frame)

                    if "detection" in self.type:
                        try:
                            results = self.model.track(frame, persist=True, imgsz=[384,640],  show=False, **self.dict_result)                
                            for index, ctr in enumerate(self.counter):
                                frame = ctr.start_counting(frame, results, index) 
                        except Exception as e:                                
                            traceback.print_exception(type(e), e, e.__traceback__)    

                    if "segmentation" in self.type:                       
                        frame=self.seg.visualize_results_usual_yolo_inference(
                            frame,
                            # self.model,
                            # imgsz=720,
                            imgsz=[384,640],
                            conf=0.35,
                            iou=0.7,
                            segment=True,
                            thickness=2,
                            show_boxes=False,
                            fill_mask=True,
                            alpha=0.9,
                            random_object_colors=False,
                            # list_of_class_colors=[(231, 64, 50),(231, 64, 50),(231, 64, 50),(231, 64, 50),(231, 64, 50),(231, 64, 50),(231, 64, 50),(231, 64, 50),(231, 64, 50)],
                            show_class=False,
                            dpi=150,
                        return_image_array=True,
                        
                            )
                    if "faceDetection" in self.type:      

                        img = Image.fromarray(frame, 'RGB')
                        results = self.model.track(img, persist=True, device=0,  imgsz=[384,640],  show=False, **self.dict_result) 

                        boxes=results[0].boxes.xyxy.cpu()               
                        faces=self.mtcnn.extract(img, boxes, save_path=None)

                        
                      
                        
                        # boxes, probs = self.model.detect(img, landmarks=False)
                        # faces=self.model.extract(img, boxes, save_path=None)
                        if faces is None or len(faces) == 0:
                            return frame
                        
                        faces=faces.to(self.device)
                        for face, box in zip(faces, boxes):
                                                        # face = Image.fromarray(face)
                            embedding = self.resnet(face.unsqueeze(0))
                            if embedding is None:
                                raise ValueError("No face detected in the target image.")

                            highest_similarity = float('-inf')
                            most_similar_image_path = None

                            embedding=embedding.to(self.device)
                            

    
                            for index, candidate_emb in enumerate(self.candidate_image_embs):  
                                # candidate_emb=candidate_emb.to(self.device)
                                similarity = torch.nn.functional.cosine_similarity(candidate_emb, embedding).item()
                                if similarity > highest_similarity:
                                    highest_similarity = similarity
                                    most_similar_image_path = self.candidate_image_face[index]
                            
                            if highest_similarity < 0.4:
                                return frame


                            frame=self.object_counter.drawBoxes(frame, box, f"{most_similar_image_path}:{highest_similarity * 100:.0f}%" )

                            if most_similar_image_path is None:
                                raise ValueError("No faces detected in the candidate images.")

                          


                        # most_similar_image, similarity_score = self.find_most_similar(frame, self.countImg)
                        # print(f"The most similar image is: {most_similar_image}")
                        # print(f"Similarity score: {similarity_score * 100:.2f}%")
                        return frame

                    
                    return frame
                else:
                    return None
      
      
    
    def restart(self):          
        if self.source is not None:
            self.source.stop()
            self.source=None
            if not (self.__queue is None):
                while not self.__queue.empty():
                    try:
                        self.__queue.get_nowait()
                        # print("after restart "+str(self.__queue.qsize()))
                    except queue.Empty:
                        continue
                    self.__queue.task_done()
        print("cae en restart")

    def stop(self):
        self.restart()        
        print("cae en stop")

    def between_callback(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        loop.run_until_complete(self.service())
        loop.close()

    def readFrame(self, wait):  
        self.wait=wait
        size=self.__queue.qsize() 
        print("size "+str(self.fps), size, self.size_prev)

        if self.source is not None:
            if self.countImg % self.stride == 0:
                if size >= 50 and size >= self.size_prev and self.fps <= 30/self.stride:
                    self.fps=self.fps* (1 + size*0.1/96)
                if size < 50 and size < self.size_prev :
                    if size==0:
                        size=1
                    self.fps=self.fps*(1 - size*0.1/50)  
        
        if self.source is None or size==0:
            self.fps=30/self.stride
            return self.black_frame, self.fps
        else:
            self.lastImage=self.__queue.get()

        if self.countImg % self.stride == 0:
            self.size_prev=size

        return self.lastImage, self.fps

    async def service (self):    
       
        VIDEO_CLOCK_RATE = 90000
             
        while True:
            try:    
                # self.fps_prev=self.fps
                
                    # if self.fps_prev >newfps:
                    #     self.fps=self.fps*0.99

                if self.__queue.qsize() >92:
                    await asyncio.sleep(self.wait)

                frame=self.read()
                if frame is None:
                    frame=self.black_frame                         
                self.__queue.put(frame)
                

            except Exception as e:          
                # self.running=False
                LOGGER.error("An exception occurred in service : %s" % e)     
                traceback.print_exception(type(e), e, e.__traceback__)
                frame=self.black_frame
                self.__queue.put(frame)


    

            