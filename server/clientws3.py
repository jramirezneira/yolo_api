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
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from collections import  Counter
from torchvision.transforms import ToPILImage, Grayscale, ToTensor, Resize
# from neuraspike import EmotionNet
import torch.nn.functional as nnf
import torch.nn as nn
import numpy as np
import torchvision
import dlib
from model.model import Mini_Xception
from face_alignment.face_alignment import FaceAlignment
from face_detector.face_detector import DnnDetector, HaarCascadeDetector
import torch.nn.functional as F




# print("CUDA disponible:", dlib.DLIB_USE_CUDA)
# print("Cantidad de GPUs:", dlib.cuda.get_num_devices())
import cv2
import numpy as np
import dlib






def transform_batch(images):  # images: Tensor de forma (batch, C, H, W)
    # Convertir a escala de grises manteniendo 3 canales
    # images = F.rgb_to_grayscale(images, num_output_channels=3)  
    # Convertir a escala de grises replicando el canal en 3 canales
    grayscale = images[:, 0:1, :, :] * 0.2989 + images[:, 1:2, :, :] * 0.587 + images[:, 2:3, :, :] * 0.114
    images = grayscale.repeat(1, 3, 1, 1)  # Convertimos de (B, 1, H, W) a (B, 3, H, W)
    

    # Redimensionar a 224x224
    images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)

    # Normalizar con los valores de ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    images = (images - mean) / std  # Normalización en batch
    
    return images

def get_label_emotion(label : int) -> str:
    label_emotion_map = { 
        0: 'Angry',
        1: 'Disgust', 
        2: 'Fear', 
        3: 'Happy', 
        4: 'Sad', 
        5: 'Surprise', 
        6: 'Neutral'        
    }
    return label_emotion_map[label]

def histogram_equalization(image):
    # image = (image*255).astype(np.uint8)
    equalized = cv2.equalizeHist(image)
    # cv2.imshow('h',equalized)
    # cv2.waitKey(0)
    # return (equalized/255).astype(np.float32)
    return equalized



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
        # self.mtcnn =MTCNN(image_size=160,margin=10, keep_all=True, select_largest=False, post_process=False)
       
        self.resnet, self.modelAge, self.vid_writer = None, None, None     
     
        self.candidate_image_embs=[]
        self.candidate_image_face=[]
        self.idsFaceRecognition=[]

        # dictionary mapping for different emotions outputs
        # self.emotion_output_dict = {0: "Angry", 1: "Fearful", 2: "Happy", 3: "Neutral",
        #                 4: "Sad", 5: "Surprised"}
        
        # dictionary mapping for different race outputs
        self.faces_race_output_dict = {0: "White", 1: "Black", 2: "Latino_Hispanic", 3: "East Asian",
                        4: "Southeast Asian", 5: "Indian", 6: "Middle Eastern"}
        
        # dictionary mapping for different age outputs
        self.faces_age_output_dict = {0: "0-2", 1: "3-9", 2: "10-19", 3: "20-29",
                        4: "30-39", 5: "40-49", 6: "50-59", 7: "60-69", 8: "70+"}
        

        self.faces_gender_output_dict= {0: "Male", 1: "Female"}

        # self.sp = dlib.shape_predictor('dlib_models/shape_predictor_68_face_landmarks.dat')
        self.sp5 = dlib.shape_predictor('dlib_models/shape_predictor_5_face_landmarks.dat')
        # self.cnn_face_detector = dlib.cnn_face_detection_model_v1('dlib_models/mmod_human_face_detector.dat')
         

        self.face_detector = DnnDetector('face_detector')
        self.face_alignment = FaceAlignment()
        
        

    def collate_fn(self, x):
        return x[0]
        
    def get_embedding_and_face(self, image, name):
        """Load an image, detect the face, and return the embedding and face."""       
        # faces = self.mtcnn(image, return_prob=False)    
        # boxes, probs = self.mtcnn.detect(image, landmarks=False)  
        img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(frame_rgb, 'RGB')
        # img = img.copy()

        faces =[]
        boxes = self.face_detector.detect_faces(img)
        # fcs= self.cnn_face_detector(img)
        # s, probs, landmarks = self.mtcnn.detect(img, landmarks=True)
        cont=0
       
        for box in boxes:
            scale=1.2
            cont=cont+1
            x, y, w, h = box
            # x = int(x1)
            # y = int(y1)
            # w = int(x2 - x1)
            # h = int(y2 - y1)
            # cx = x + w // 2
            # cy = y + h // 2
            # new_w = int(w * 1.5)
            # new_h = int(h * 1.5)
            # new_x = int(cx - new_w // 2)
            # new_y = int(cy - new_h // 2)
            # rect = dlib.rectangle(left=int(x1), top=int(y1), right=int(x2), bottom=int(y2))
            # print(type(box))
            # (x_min, y_min, x_max, y_max) = box
            # x, y, width, height = box
            # rect = face.rect
            # x = rect.left()
            # y = rect.top()
            # w = rect.width()
            # h = rect.height()
            # box =(rect.left(), rect.top(), rect.right(), rect.bottom())   
            # box=(x,y,w,h)
        # for box in boxes:    
        #     (x,y,w,h) = box
            # dlib_box = dlib.rectangle(face.rect)
            dlib_box=dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)
            
            landmarks=self.sp5(img, dlib_box)

            # img = Image.fromarray(img)
            # preprocessing
            # landmarks = self.detector(img, box)
            # input_face = self.face_alignment.frontalize_face((new_x, new_y, new_w, new_h), img)
            input_face = self.face_alignment.frontalize_face(box,landmarks, img)
            
            input_face = cv2.resize(input_face, (160,160))
            # input_face = histogram_equalization(input_face)
            
                          
            # input_face = histogram_equalization(input_face)
            # cv2.imshow('input face', cv2.resize(input_face, (160, 160)))

            input_face = transforms.ToTensor()(input_face).to(self.device)
            faces.append(input_face)

        # faces=self.mtcnn.extract(image, boxes, save_path=None).to(self.device) 

        if faces is None or len(faces) == 0:
            return None, None
        
        # faces[0].to(self.device)

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

            cont =0
            for x, y in loader:
                candidate_emb, candidate_face = self.get_embedding_and_face(x, dataset.idx_to_class[y] +str(cont))
                if candidate_emb is None:
                    continue
                else:
                    self.candidate_image_embs.append(candidate_emb)
                    self.candidate_image_face.append(dataset.idx_to_class[y])
                cont=cont+1
     
            self.idsFaceRecognition=[]

            

            # load the emotionNet weights
            # self.modelEmoc = EmotionNet(num_of_channels=1, num_of_classes=len(self.emotion_output_dict))
            # model_weights = torch.load("model.pth")
            # self.modelEmoc.load_state_dict(model_weights)
            # self.modelEmoc.to(self.device)
            # self.modelEmoc.eval()
            self.mini_xception = Mini_Xception().to(self.device)
            self.mini_xception.eval()

            # Load model
            checkpoint = torch.load('model/checkpoint/model_weights/weights_epoch_43.pth.tar', map_location=self.device)
            self.mini_xception.load_state_dict(checkpoint['mini_xception'])

            self.model_fair_7 = torchvision.models.resnet34(pretrained=True)
            self.model_fair_7.fc = nn.Linear(self.model_fair_7.fc.in_features, 18)
            self.model_fair_7.load_state_dict(torch.load('res34_fair_align_multi_7_20190809.pt'))
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
        # self.source = CamGear(source="https://www.youtube.com/watch?v=KHUCDF2FQWU",  stream_mode=True if self.SourceType=="yt" else False,  logging=False, **options if self.SourceType=="yt" else {}).start()    
        self.source = CamGear(source="https://www.youtube.com/watch?v=dnfADeXIi8A",  stream_mode=True if self.SourceType=="yt" else False,  logging=False, **options if self.SourceType=="yt" else {}).start()    
        # self.source = CamGear(source=0).start() 
        self.vid_writer = cv2.VideoWriter(
                "example2.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, (640, 360)
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
                        (img, boxes)=frame
                        frame=img
                        # cv2.imwrite('camara1.jpg', frame)
                        
                        # img = Image.fromarray(frame_rgb, 'RGB')
                        # img = img.copy()

                        facesAgeList =[]
                        facesEmocList =[]
                        faceRecognitionList=[]
                        facesBoxes =[]
                        line_thickness=1                                
                        color=(0,130,0)

                        

                        facesAgeList =[]
                        facesEmocList =[]
                        esfrontal=[]

                        # boxes = self.face_detector.detect_faces(img)
                        # faces= self.cnn_face_detector(img, 1)
                        # faces, probs, landmarks = self.mtcnn.detect(img, landmarks=True)
                        if len(boxes) == 0:
                            return frame
                        # np_img= np.array(img)
                        landmarksArr = dlib.full_object_detections()
                        listEmocs=[]
                        listfacestest=[]

                        for box in boxes:
                            x, y, w, h = box
                            facesBoxes.append(box)
                            dlib_box=dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)                            
                            # shape=self.sp(img, dlib_box)
                            shape5=self.sp5(img, dlib_box)
                           

                            # landmarks68 = np.array([[p.x, p.y] for p in shape.parts()])

                            # lf=landmarks68[16][0]-landmarks68[28][0]
                            # rg=landmarks68[28][0]-landmarks68[0][0]

                            # sumEdges=lf+rg

                            # if lf/sumEdges < 0.4 or rg/sumEdges < 0.4:
                            #     esfrontal.append(f"noooo es frontal {lf/sumEdges:.2f} {rg/sumEdges:.2f} {landmarks68[28][0]} {landmarks68[16][0]} {landmarks68[0][0]}")
                            # else:
                            #     esfrontal.append(f"fes frontal {lf/sumEdges:.2f} {rg/sumEdges:.2f} {landmarks68[28][0]} {landmarks68[16][0]} {landmarks68[0][0]}")  

                            esfrontal.append("")
                            # landmarks5 = np.array([
                            #     landmarks68[45],
                            #     landmarks68[42],                                
                            #     landmarks68[36],                               
                            #     landmarks68[39],
                            #     landmarks68[33]
                            # ])
                            # Convertir los puntos a dlib.point
                            # points_dlib = [dlib.point(int(x), int(y)) for (x, y) in landmarks5]

                            # Crear el full_object_detection con 5 puntos
                            # landmarks5 = dlib.full_object_detection(dlib_box, points_dlib)
                            # points_dlib68 = [dlib.point(int(x), int(y)) for (x, y) in landmarks68]
                            # for i, p in enumerate(points_dlib68):
                            #     cv2.circle(img, (p.x, p.y), 3, (0, 255, 0), -1)
                                # cv2.putText(img, str(i), (p.x, p.y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),4)
                            # cv2.imshow("Debug", img)

                            

                            landmarksArr.append(shape5)
                            facePreproc = self.face_alignment.frontalize_face(box, shape5, img)
                            listfacestest.append(facePreproc)
                            faceEmoc = cv2.cvtColor(facePreproc, cv2.COLOR_BGR2GRAY)
                            faceEmoc = cv2.resize(faceEmoc, (48,48))
                          
                            faceEmoc = histogram_equalization(faceEmoc)
                            # cv2.imshow('input face', faceEmoc)

                            faceRecognition = cv2.resize(facePreproc, (160,160))
                            faceRecognition = transforms.ToTensor()(faceRecognition)
                            faceRecognitionList.append(faceRecognition)

                            faceEmoc = transforms.ToTensor()(faceEmoc)
                            facesEmocList.append(faceEmoc)

                        if landmarksArr:
                            
                            faces_chips = dlib.get_face_chips(img, landmarksArr, size=300, padding = 0.25)
                            # listEmocs = detect_emotion(faces_chips,True)
                            # print(listEmocs)
                        else:
                            return frame

                  

                        with torch.no_grad():
                            facesAgeList= [torch.from_numpy(chip).permute(2, 0, 1).float() / 255.0 for chip in faces_chips]
                            facesAgeList = torch.stack(facesAgeList, dim=0).to(self.device)
                            facesAgeList= transform_batch(facesAgeList)
                            facesEmocList = torch.stack(facesEmocList, dim=0).to(self.device)
                            faceRecognitionList = torch.stack(faceRecognitionList, dim=0).to(self.device)
                            
                            predictionsAgegender = self.model_fair_7(facesAgeList).detach().cpu()
                            predictionsEmoc = self.mini_xception(facesEmocList).detach().cpu()
                            embeddings = self.resnet(faceRecognitionList).detach().cpu().to(self.device)

                            for  box, age, emoc, emb, frt in zip(facesBoxes, predictionsAgegender, predictionsEmoc, embeddings, esfrontal):
                                torch.set_printoptions(precision=6)
                                softmax = torch.nn.Softmax()
                                emotions_soft = softmax(emoc.squeeze()).reshape(-1,1).cpu().detach().numpy()
                                emotions_soft = np.round(emotions_soft, 3)

                                emotion = torch.argmax(emoc)                
                                percentage = round(emotions_soft[emotion].item()*100)
                                emotion = emotion.squeeze().cpu().detach().item()
                                emotion = get_label_emotion(emotion)

                                age = age.cpu().detach().numpy()
                                age = np.squeeze(age)

                                race_outputs = age[:7]
                                gender_outputs = age[7:9]
                                age_outputs = age[9:18]

                                race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
                                gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
                                age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

                                race_pred = self.faces_race_output_dict[np.argmax(race_score)]
                                gender_pred = self.faces_gender_output_dict[np.argmax(gender_score)]
                                age_pred = self.faces_age_output_dict[np.argmax(age_score)]
                                prob_text = f"{emotion}: {percentage}%"

                                labelOutput = None
                                for index, candidate_emb in enumerate(self.candidate_image_embs):                                                          
                                    distance = (emb-candidate_emb).norm().item()
                                    if distance < 0.95:  
                                        color=(255,0,0)  
                                        labelOutput = self.candidate_image_face[index]    
                                        break
                                if labelOutput is None:
                                    labelOutput="Unknown"
                                    color=(0,0,255)
                                    line_thickness=1
                                # cv2.putText(frame, emotion, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200))
                                # cv2.putText(frame, str(percentage), (x + w - 40,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                #              (200,200,0))
                                # cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 3)

                                frame=self.object_counter.drawBoxes(img, box, f"{labelOutput}",[f"{prob_text}", f"{gender_pred}",f"{age_pred}",f"{race_pred}"], color=color,  line_thickness=line_thickness ) 

                        # cv2.putText(frame, str(25), (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
                        # cv2.imshow("Video", frame)   
                        # if cv2.waitKey(1) & 0xff == 27:
                           
                        #     break

                        return frame
            else:
                return None
            
    def get_size(self, img):
        if isinstance(img, (np.ndarray, torch.Tensor)):
            return img.shape[1::-1]
        else:
            return img.size
      
    # def transform(self, image):
    #     """Transform input face image for the model."""
    #     return transforms.Compose(
    #         [
    #             transforms.ToPILImage(),
    #             transforms.Resize(64),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #         ]
    #     )(image)
      
    
    def restart(self):          
        if self.source is not None:
            self.source.stop()
            self.vid_writer.release()
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
            self.vid_writer.write(self.lastImage)

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


    

            