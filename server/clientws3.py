# import necessary libs
import  cv2
from vidgear.gears import CamGear
from vidgear.gears.helper import create_blank_frame, reducer, retrieve_best_interpolation
from utils.solutions import object_counter
from utils.general import image_resize, getConfProperty
from urllib.parse import urlparse
import cv2
import torch
from ultralytics.utils import LOGGER
from server.patched_yolo_infer.functions_extra import Segment_Stream_Class
import traceback
import threading
import asyncio
import time
import numpy as np
import queue
import uvicorn
from vidgear.gears.asyncio import WebGear_RTC
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image
from torchvision import transforms as T
from collections import  Counter
from torchvision.transforms import ToPILImage
from torchvision.transforms import Grayscale
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision import transforms
from neuraspike import EmotionNet
import torch.nn.functional as nnf
import torch.nn as nn
import numpy as np
import torchvision


color=(255,0,0)
line_thickness=2

trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

data_transform = transforms.Compose([
    ToPILImage(),
    Grayscale(num_output_channels=1),
    Resize((48, 48)),
    ToTensor()
])

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
        self.mtcnn =MTCNN(image_size=160, keep_all=True, select_largest=False, post_process=True)
       
        self.resnet, self.modelAge, self.vid_writer = None, None, None     
     
        self.candidate_image_embs=[]
        self.candidate_image_face=[]
        self.idsFaceRecognition=[]


    def collate_fn(self, x):
        return x[0]
        
    def get_embedding_and_face(self, image):
        """Load an image, detect the face, and return the embedding and face."""       
        faces = self.mtcnn(image, return_prob=False)       

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
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

            # self.modelAge = Model().to(self.device)
            # self.modelAge.load_state_dict(torch.load("weights.pt", map_location="cpu"))

            dataset = datasets.ImageFolder('database')
            dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
            loader = DataLoader(dataset, collate_fn=self.collate_fn, num_workers=0)

            self.candidate_image_embs=[]
            self.candidate_image_face=[]

            for x, y in loader:
                candidate_emb, candidate_face = self.get_embedding_and_face(x)
                if candidate_emb is None:
                    continue
                else:
                    self.candidate_image_embs.append(candidate_emb)
                    self.candidate_image_face.append(dataset.idx_to_class[y])
     
            self.idsFaceRecognition=[]

            # dictionary mapping for different outputs
            self.emotion_dict = {0: "Angry", 1: "Fearful", 2: "Happy", 3: "Neutral",
                            4: "Sad", 5: "Surprised"}

            # load the emotionNet weights
            self.modelEmoc = EmotionNet(num_of_channels=1, num_of_classes=len(self.emotion_dict))
            model_weights = torch.load("42-bestmodel.pt")
            self.modelEmoc.load_state_dict(model_weights)
            self.modelEmoc.to(self.device)
            self.modelEmoc.eval()

            self.model_fair_7 = torchvision.models.resnet34(pretrained=True)
            self.model_fair_7.fc = nn.Linear(self.model_fair_7.fc.in_features, 18)
            self.model_fair_7.load_state_dict(torch.load('res34_fair_align_multi_4_20190809.pt'))
            self.model_fair_7 = self.model_fair_7.to(self.device)
            self.model_fair_7.eval()


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

        if urlparse(self.sourceVideo).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):
            self.SourceType="yt"      
        else:
            self.SourceType="rtsp"        

        options = {"STREAM_RESOLUTION": "360p", "CAP_PROP_FRAME_WIDTH":640, "CAP_PROP_FRAME_HEIGHT":360 }
        # self.source = CamGear(source="https://www.youtube.com/watch?v=3yEubbOTof0",  stream_mode=True if self.SourceType=="yt" else False,  logging=False, **options if self.SourceType=="yt" else {}).start()    
        self.source = CamGear(source="https://www.youtube.com/watch?v=wz_42pckM7w",  stream_mode=True if self.SourceType=="yt" else False,  logging=False, **options if self.SourceType=="yt" else {}).start()    
        
        self.vid_writer = cv2.VideoWriter(
                "example.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360)
            )
        

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
                return None
            self.countImg= self.countImg+1
            # print(self.countImg)
            frame =  self.source.read()     
            # path = os.path.dirname(os.path.realpath(__file__))
            # cv2.imwrite(os.path.join(path , 'Town of Collingwood.jpg'), frame)
            if frame is not None:
                if self.countImg % self.stride == 0:   
                # check if frame is available
                                    
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
                        results = self.model.track(img, device=0, persist=True, imgsz=[384,640],  show=False, **self.dict_result)  
                        boxes=results[0].boxes.xyxy.cpu()  
                        
                        if results is None or results[0].boxes.id is None:
                            return frame
                      
                        
                        faces=self.mtcnn.extract(img, boxes, save_path=None).to(self.device)
                        track_ids=results[0].boxes.id.int().cpu().tolist()                       
                        

                        facesList =[]  
                        boxesList=[]
                        track_idsList=[]
                        facesEmocList =[]
                        facesAgeList =[]
                        track_idsEmocList=[]
                        embeddings=[]
                        # genders, ages= None, None

                        if boxes is not None:
                            for track_id, box, face in zip(track_ids, boxes, faces):

                                if Counter([p[0] for p in self.idsFaceRecognition])[track_id] <=20:
                                    facesList.append(face)  
                                    boxesList.append(box)
                                    track_idsList.append(track_id)

                                (start_x, start_y, end_x, end_y) = np.asarray(box, dtype=int)                                
                                faceEmoc = frame[start_y:end_y, start_x:end_x]
                                # faceAge = frame[start_y:end_y, start_x:end_x]
                                faceEmoc = data_transform(faceEmoc)
                                faceAge= trans(face)
                                facesEmocList.append(faceEmoc)
                                facesAgeList.append(faceAge)
                                track_idsEmocList.append(track_id)

                            # if facesEmocList:
                            facesEmocList = torch.stack(facesEmocList, dim=0).to(self.device)
                            facesAgeList = torch.stack(facesAgeList, dim=0).to(self.device)
                            predictionsEmoc = self.modelEmoc(facesEmocList).detach().cpu().to(self.device)
                            predictionsAgegender = self.model_fair_7(facesAgeList).detach().cpu().to(self.device)
                            # genders, ages = self.modelAge(facesAgeList)
                            
                              

                            if facesList:
                                facesList = torch.stack(facesList, dim=0).to(self.device)
                                boxesList = torch.stack(boxesList, dim=0).to(self.device)
                                embeddings = self.resnet(facesList).detach().cpu().to(self.device)
                                # track_idsList = torch.stack(track_idsList, dim=0).to(self.device)
                            # else:
                            #     return frame
                            # _, bs, c, h, w=aligned.shape
                            # aligned=aligned.reshape(bs, c, h, w)

                        # face_images = torch.stack(aligned)
                        #     genders, ages = self.modelAge(aligned)
                        #     genders = torch.round(genders)
                        #     ages = torch.round(ages).long()

                            # for i, box in enumerate(boxes):
                            #     frame=self.object_counter.drawBoxes(frame, box, f"{'Man' if genders[i] == 0 else 'Woman'}: {ages[i].item()} years old" ) 

                            

                            
                        # print(facesList.shape)
                        
                        
                        # print(facesList.shape)
                        # facesList = data_transform(facesList)
                                
                            predictionsEmoc = nnf.softmax(predictionsEmoc, dim=1)
                            # top_p, top_class = prob.topk(1, dim=1)
                            # top_p, top_class = top_p.item(), top_class.item()
                            # emotion_prob = [p.item() for p in prob[0]]
                            # emotion_value = self.emotion_dict.values()

                            # genders = torch.round(genders)
                            # ages = torch.round(ages).long()

                        
                            for trk_id, box, predEmoc, age in zip(track_ids, boxes, predictionsEmoc, predictionsAgegender):

                                age = age.cpu().detach().numpy()
                                age = np.squeeze(age)

                                race_outputs = age[:7]
                                gender_outputs = age[7:9]
                                age_outputs = age[9:18]

                                race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
                                gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
                                age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

                                race_pred = np.argmax(race_score)
                                gender_pred = np.argmax(gender_score)
                                age_pred = np.argmax(age_score)

                                # race_scores_fair.append(race_score)
                                # gender_scores_fair.append(gender_score)
                                # age_scores_fair.append(age_score)

                                # race_preds_fair.append(race_pred)
                                # gender_preds_fair.append(gender_pred)
                                # age_preds_fair.append(age_pred)


                                labelOutput = None
                                prob_text=""
                                line_thickness=1                                
                                color=(0,130,0)

                                top_p, top_class = predEmoc.topk(1, dim=0)
                                top_p, top_class = top_p.item(), top_class.item()

                                # emotion_prob = [p.item() for p in predEmoc]
                                # emotion_value = self.emotion_dict.values()
                                # if top_p > 0.5:
                                face_emotion = self.emotion_dict[top_class]
                                prob_text = f"{face_emotion}: {top_p * 100:.0f}%"
                                    # print(prob_text)
                                # for (i, (emotion, prob)) in enumerate(zip(emotion_value, emotion_prob)):
                                #     prob_text = f"{emotion}: {prob * 100:.2f}%"
                                #     width = int(prob * 300)
                                #     print(prob_text)
                                
                                

                                if Counter([p[0] for p in self.idsFaceRecognition])[trk_id] >15:
                                    # indx=track_ids.index(trk_id) 
                                    # box= boxes[indx]
                                    labelOutput=f"{[p[1] for p in self.idsFaceRecognition if p[0] == trk_id][0]}"
                                else:
                                    if trk_id in track_idsList: 
                                        indx=track_idsList.index(trk_id) 
                                        emb= embeddings[indx]                      
                                        # box= boxesList[indx] 
                                        for index, candidate_emb in enumerate(self.candidate_image_embs):                                                          
                                            distance = (emb-candidate_emb).norm().item()
                                            if distance < 0.95:  
                                                color=(255,0,0)  
                                                # print(Counter([p[0] for p in self.idsFaceRecognition])[track_id])
                                                
                                                # print(f"{track_id}:{distance}")
                                                labelOutput = self.candidate_image_face[index]                                            
                                                self.idsFaceRecognition.append([trk_id,labelOutput])
                                                break
                                        if labelOutput is None:
                                            labelOutput="Unknown"
                                            color=(0,0,255)
                                            line_thickness=1
                                        
                                        
                                
                                frame=self.object_counter.drawBoxes(frame, box, f"{labelOutput}",f"{prob_text}", color=color,  line_thickness=line_thickness ) 
                            
                         
                            
                            # for index, candidate_emb in enumerate(self.candidate_image_embs):
                            #     most_similar_image_path = None
                                
                            #     for emb, box in zip(embeddings, boxes):                      
                            #         distance = (emb-candidate_emb).norm().item()
                            #         bx=box
                            #         if distance < 1.0:
                            #             most_similar_image_path = self.candidate_image_face[index]
                            #             break
                            #     if most_similar_image_path is None:
                            #         most_similar_image_path="Unknown person"
                            
                            #     frame=self.object_counter.drawBoxes(frame, bx, f"{most_similar_image_path}", color=(0,255,0) if  most_similar_image_path is None else (255,0,0)  ) 
                    return frame
            else:
                return None
      
    def transform(self, image):
        """Transform input face image for the model."""
        return T.Compose(
            [
                T.ToPILImage(),
                T.Resize(64),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )(image)
      
    
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
        print("size "+str(round(self.fps)), size, self.size_prev)

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

        return self.lastImage, round(self.fps)

    async def service (self):    
       
        VIDEO_CLOCK_RATE = 90000
             
        while True:
            try:    
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


    

            