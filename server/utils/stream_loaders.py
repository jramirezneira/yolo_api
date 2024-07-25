# Ultralytics YOLO 🚀, AGPL-3.0 license

import glob
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
import torch
from PIL import Image

from ultralytics.data.augment import LetterBox
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.utils import LOGGER, ROOT, is_colab, is_kaggle, ops
from ultralytics.utils.checks import check_requirements
import threading
from utils.general import image_resize, getConfPropertie, setStatus
import time
import subprocess
from subprocess import Popen
import asyncio
import multiprocessing
from ultralytics import YOLO
from ultralytics.solutions import object_counter


@dataclass
class SourceTypes:
    webcam: bool = False
    screenshot: bool = False
    from_img: bool = False
    tensor: bool = False


class LoadStreamNoThread:
    def __init__(self, source):
        names = {   0: 'person',  1: 'bicycle',  2: 'car',  3: 'motorcycle',  4: 'airplane',  5: 'bus',  6: 'train',  7: 'truck',
            8: 'boat',  9: 'traffic light',  10: 'fire hydrant',  11: 'stop sign',  12: 'parking meter',  13: 'bench',  14: 'bird',
            15: 'cat',  16: 'dog',  17: 'horse',  18: 'sheep',  19: 'cow',  20: 'elephant',  21: 'bear',  22: 'zebra',  23: 'giraffe',
            24: 'backpack',  25: 'umbrella',  26: 'handbag',  27: 'tie',  28: 'suitcase',  29: 'frisbee',  30: 'skis',  31: 'snwboard',
            32: 'sports ball',  33: 'kite',  34: 'baseball bat',  35: 'baseball glove',  36: 'skateboard',  37: 'surfboard',  38: 'tennis racket',
            39: 'bottle',  40: 'wine glass',  41: 'cup',  42: 'fork',  43: 'knife',  44: 'spoon',  45: 'bowl',  46: 'banana',  47: 'apple',
            48: 'sandwich',  49: 'orange',  50: 'broccoli',  51: 'carrot',  52: 'hot dog',  53: 'pizza',  54: 'donut',  55: 'cake',
            56: 'chair',  57: 'couch',  58: 'potted plant',  59: 'bed',  60: 'dining table',  61: 'toilet',  62: 'tv',  63: 'laptop',
            64: 'mouse',  65: 'remote',  66: 'keyboard',  67: 'cell phone',  68: 'microwave',  69: 'oven',  70: 'toaster',  71: 'sink',
            72: 'refrigerator',  73: 'book',  74: 'clock',  75: 'vase',  76: 'scissors',  77: 'teddy bear',  78: 'hair drier',  79: 'toothbrush'}




        self.cmd = 'python3 stream_rtsp_server.py'
        self.thrP = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO("yolov8n.pt").to(self.device)

        self.counter=[]
        region_points, self.stride =getConfPropertie("region_points", "stride")
        region_points_dict = [x for x in region_points if x['source'] == source and x['available'] == 1][0]

        for i, rp in enumerate(region_points_dict["region_points"]):
            ctr= object_counter.ObjectCounter()
            ctr.set_args(view_img=False,
                        reg_pts=rp,
                        classes_names=names,
                        draw_tracks=True,
                        reg_counts=region_points_dict["reg_counts"][i]
                        )
            self.counter.append(ctr)




        if urlparse(source).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):
            check_requirements(('pafy', 'youtube_dl==2020.12.2'))
            # import pafy
            # source = pafy.new(source).getbest(preftype='mp4').url   
            cmd="python3 stream_rtsp_server.py"
            self.thr = threading.Thread(target=self.startStreamRtspServer, args=(), kwargs={})
            self.thr.start()  
            # self.proc = multiprocessing.Process(target=self.startStreamRtspServer, args=())
            # self.proc.start()
            source="rtsp://127.0.0.1:8554/video_stream"
        self.cv2= cv2       

        self.cap=self.openStreamRtspServer(source)               
        self.cap.set(self.cv2.CAP_PROP_BUFFERSIZE,500)
        if not self.cap.isOpened():
                raise ConnectionError(f'Failed to open')
        
        (major_ver, minor_ver, subminor_ver) = (self.cv2.__version__).split('.') 
        if int(major_ver)  < 3 :
            fps = self.cap.get(self.cv2.cv.CV_CAP_PROP_FPS)
            print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
        else :
            fps = self.cap.get(self.cv2.CAP_PROP_FPS)
            print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

        return None
    
    def startStreamRtspServer(self):        
        self.proc = subprocess.Popen("python3 stream_rtsp_server.py", stdout=subprocess.PIPE, shell=True)
        out, err = self.proc.communicate() 
        result = out.split('\n')
        for lin in result:
            if not lin.startswith('#'):
                print(lin)

    # def stopStreamRtspServer(self):
    #     subprocess.Popen.terminate(self.cmd)

    def openStreamRtspServer(self, source):
        while True:
            try:
                video=cv2.VideoCapture(source)
                if video is None or not video.isOpened():
                    raise ConnectionError
                else:
                    return video
            except ConnectionError:
                    print("Retrying connection to ",source," in ",str(0.5), " seconds...")
                    time.sleep(0.5)

       


    # # read frames as soon as they are available, keeping only most recent one
    # def _reader(self):
    #     while self.cap.isOpened:
    #         time.sleep(0.01)
           
    #         if(self.q.qsize()<=300): 
    #             self.cap.grab() 
    #             ret, frame = self.cap.retrieve()
    #             if not ret:
    #                 break
    #             frame=image_resize(frame, height = 720)
    #             print(self.q.qsize())
    #             # if not self.q.empty():
    #             #     try:
    #             #         self.q.get()   # discard previous (unprocessed) frame
    #             #     except queue.Empty:
    #             #         pass
    #             self.q.put(frame)

    def read(self):
        return self.q.get()
    
    def startPrediction(self, server):
        self.thrP = threading.Thread(target=self.startStreamRtspServer, args=( [server]), kwargs={})
        self.thrP.start()  


        # import pafy  # noqa
        # source = pafy.new(source).getbest(preftype='mp4').url
        # self.cap = cv2.VideoCapture(source)
        # assert self.cap.isOpened(), "Error reading video file"
        # return None

    def getCap(self):
        print("pasa 0")
        return self.cap
    
    def service(self, server):    
       
    
        n=0
        while self.cap.isOpened:        
            try:
                time.sleep(0.00000001)
                n += 1        
            
                # cap.read()
                # cap.set(cv2.CAP_PROP_FPS,25) 
                success = self.cap.grab() 
                if not success: break                      
                
                results=None
                if n % self.stride== 0:
                    ret, im0 = self.cap.retrieve()
                    if not ret:
                        break
                    im0=image_resize(im0, height = 720)
                    dict_result=dict()
                    dict_result["verbose"] =False
                    results = self.model.track(im0, persist=True, imgsz=640, show=False, **dict_result)

                
                    for ctr in self.counter:
                        im0 = ctr.start_counting(im0, results)  
                # if isStreaming:
                    server.send(im0)        
                    # else:
                    #     yield (b'--frame\r\n'
                    #             b'Content-Type: image/jpeg\r\n\r\n' + im0 + b'\r\n')
            except Exception as e:
                LOGGER.error("Error in while read : %s" % e)
                continue

        cv2.destroyAllWindows()
        setStatus("offline")


class LoadStreams:
    
    # YOLOv8 streamloader, i.e. `yolo predict source='rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources='file.streams', imgsz=640, stride=32, auto=True, transforms=None, vid_stride=1):
        """Initialize instance variables and check for consistent input stream shapes."""
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.mode = 'stream'
        self.imgsz = imgsz
        self.stride = stride
        self.vid_stride = vid_stride  # video frame-rate stride
        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.sources = [ops.clean_str(x) for x in sources]  # clean source names for later
        self.imgs, self.fps, self.frames, self.threads, self.cap = [None] * n, [0] * n, [0] * n, [None] * n, [None] * n
        
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            if urlparse(s).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):  # if source is YouTube video
                # YouTube format i.e. 'https://www.youtube.com/watch?v=Zgi9g1ksQHc' or 'https://youtu.be/Zgi9g1ksQHc'
                check_requirements(('pafy', 'youtube_dl'))
                import pafy  # noqa
                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            if s == 0 and (is_colab() or is_kaggle()):
                raise NotImplementedError("'source=0' webcam not supported in Colab and Kaggle notebooks. "
                                          "Try running 'source=0' in a local environment.")
            self.cap = cv2.VideoCapture(s)
            if not self.cap.isOpened():
                raise ConnectionError(f'{st}Failed to open {s}')
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames[i] = max(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback            

            success, self.imgs[i] = self.cap.read()  # guarantee first frame

            self.imgs[i]=image_resize(self.imgs[i], height = 720)

            if not success or self.imgs[i] is None:
                raise ConnectionError(f'{st}Failed to read images from {s}')
            self.threads[i] = Thread(target=self.update, args=([i,  s, 0.02 if self.frames[i] == float('inf') else 0.0]), daemon=False)
            # self.update(i, cap, s, 0.02 if self.frames[i] == float('inf') else 0.0)
            LOGGER.info(f'{st}Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)')
            self.threads[i].start()
        LOGGER.info('')  # newline

        # Check for common shapes
        s = np.stack([LetterBox(imgsz, auto, stride=stride)(image=x).shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        self.auto = auto and self.rect
        self.transforms = transforms  # optional
        self.bs = self.__len__()

        if not self.rect:
            LOGGER.warning('WARNING ⚠️ Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    # def newThreads(s):
    #      # Start thread to read frames from video stream
    #     st = f'{i + 1}/{n}: {s}... '
    #     if urlparse(s).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):  # if source is YouTube video
    #         # YouTube format i.e. 'https://www.youtube.com/watch?v=Zgi9g1ksQHc' or 'https://youtu.be/Zgi9g1ksQHc'
    #         check_requirements(('pafy', 'youtube_dl==2020.12.2'))
    #         import pafy  # noqa
    #         s = pafy.new(s).getbest(preftype='mp4').url  # YouTube URL
    #     s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
    #     if s == 0 and (is_colab() or is_kaggle()):
    #         raise NotImplementedError("'source=0' webcam not supported in Colab and Kaggle notebooks. "
    #                                     "Try running 'source=0' in a local environment.")
    #     cap = cv2.VideoCapture(s)
    #     if not cap.isOpened():
    #         raise ConnectionError(f'{st}Failed to open {s}')
    #     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
    #     self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
    #     self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback            

    #     success, self.imgs[i] = self.cap.read()  # guarantee first frame
    #     if not success or self.imgs[i] is None:
    #         raise ConnectionError(f'{st}Failed to read images from {s}')
    #     self.threads[i] = Thread(target=self.update, args=([i,  s, 0.02 if self.frames[i] == float('inf') else 0.0]), daemon=False)
    #     # self.update(i, cap, s, 0.02 if self.frames[i] == float('inf') else 0.0)
    #     LOGGER.info(f'{st}Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)')
    #     self.threads[i].start()
    #     return None

    def update(self, i,  stream, timewait):
        """Read stream `i` frames in daemon thread."""
        
        n, f = 0, self.frames[i]  # frame number, frame array
        while self.cap.isOpened() and n < f:
            try:     
                n += 1
                self.cap.grab()  # .read() = .grab() followed by .retrieve()
                if n % self.vid_stride == 0:
                    success, im = self.cap.retrieve()
                    if success:
                        self.imgs[i] = image_resize(im, height = 720)
                        time.sleep(timewait)
                    else:
                        LOGGER.warning('WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.')
                        self.imgs[i] = np.zeros_like(self.imgs[i])
                        self.cap.open(stream)  # re-open stream if signal was lost
            # time.sleep(0.02 if self.frames[i] == 'inf' else 0.0) # wait time
            except:
                LOGGER.error("Error in cv2.VideoCapture instance")
            

    def __iter__(self):
        """Iterates through YOLO image feed and re-opens unresponsive streams."""
        self.count = -1
        print("pasa 0")
        return self

    def __next__(self):
        """Returns source paths, transformed and original images for processing YOLOv5."""
        # print("pasa 1")
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            print("pasa 2")
            cv2.destroyAllWindows()
            raise StopIteration

        im0 = self.imgs.copy()
        # if self.transforms:
        #     print("pasa 3")
        #     im = np.stack([self.transforms(x) for x in im0])  # transforms
        # else:
        #     # print("pasa 4")
        #     im = np.stack([LetterBox(self.imgsz, self.auto, stride=self.stride)(image=x) for x in im0])
        #     im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        #     im = np.ascontiguousarray(im)  # contiguous

        return self.sources, None, im0, None, ''

    def __len__(self):
        """Return the length of the sources object."""
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


class LoadScreenshots:
    # YOLOv8 screenshot dataloader, i.e. `yolo predict source=screen`
    def __init__(self, source, imgsz=640, stride=32, auto=True, transforms=None):
        """source = [screen_number left top width height] (pixels)."""
        check_requirements('mss')
        import mss  # noqa

        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None  # default to full screen 0
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)
        self.imgsz = imgsz
        self.stride = stride
        self.transforms = transforms
        self.auto = auto
        self.mode = 'stream'
        self.frame = 0
        self.sct = mss.mss()
        self.bs = 1

        # Parse monitor shape
        monitor = self.sct.monitors[self.screen]
        self.top = monitor['top'] if top is None else (monitor['top'] + top)
        self.left = monitor['left'] if left is None else (monitor['left'] + left)
        self.width = width or monitor['width']
        self.height = height or monitor['height']
        self.monitor = {'left': self.left, 'top': self.top, 'width': self.width, 'height': self.height}

    def __iter__(self):
        """Returns an iterator of the object."""
        return self

    def __next__(self):
        """mss screen capture: get raw pixels from the screen as np array."""
        im0 = np.array(self.sct.grab(self.monitor))[:, :, :3]  # [:, :, :3] BGRA to BGR
        s = f'screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: '

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = LetterBox(self.imgsz, self.auto, stride=self.stride)(image=im0)
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
        self.frame += 1
        return str(self.screen), im, im0, None, s  # screen, img, original img, im0s, s


class LoadImages:
    # YOLOv8 image/video dataloader, i.e. `yolo predict source=image.jpg/vid.mp4`
    def __init__(self, path, imgsz=640, stride=32, auto=True, transforms=None, vid_stride=1):
        """Initialize the Dataloader and raise FileNotFoundError if file not found."""
        if isinstance(path, str) and Path(path).suffix == '.txt':  # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.imgsz = imgsz
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        self.bs = 1
        if any(videos):
            self.orientation = None  # rotation degrees
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        if self.nf == 0:
            raise FileNotFoundError(f'No images or videos found in {p}. '
                                    f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}')

    def __iter__(self):
        """Returns an iterator object for VideoStream or ImageFolder."""
        self.count = 0
        return self

    def __next__(self):
        """Return next image, path and metadata from dataset."""
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            for _ in range(self.vid_stride):
                self.cap.grab()
            success, im0 = self.cap.retrieve()
            while not success:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                success, im0 = self.cap.read()

            self.frame += 1
            # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            if im0 is None:
                raise FileNotFoundError(f'Image Not Found {path}')
            s = f'image {self.count}/{self.nf} {path}: '

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = LetterBox(self.imgsz, self.auto, stride=self.stride)(image=im0)
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

        return path, im, im0, self.cap, s

    def _new_video(self, path):
        """Create a new video capture object."""
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        if hasattr(cv2, 'CAP_PROP_ORIENTATION_META'):  # cv2<4.6.0 compatibility
            self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees
            # Disable auto-orientation due to known issues in https://github.com/ultralytics/yolov5/issues/8493
            # self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

    def _cv2_rotate(self, im):
        """Rotate a cv2 video manually."""
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        """Returns the number of files in the object."""
        return self.nf  # number of files


class LoadPilAndNumpy:

    def __init__(self, im0, imgsz=640, stride=32, auto=True, transforms=None):
        """Initialize PIL and Numpy Dataloader."""
        if not isinstance(im0, list):
            im0 = [im0]
        self.paths = [getattr(im, 'filename', f'image{i}.jpg') for i, im in enumerate(im0)]
        self.im0 = [self._single_check(im) for im in im0]
        self.imgsz = imgsz
        self.stride = stride
        self.auto = auto
        self.transforms = transforms
        self.mode = 'image'
        # Generate fake paths
        self.bs = len(self.im0)

    @staticmethod
    def _single_check(im):
        """Validate and format an image to numpy array."""
        assert isinstance(im, (Image.Image, np.ndarray)), f'Expected PIL/np.ndarray image type, but got {type(im)}'
        if isinstance(im, Image.Image):
            if im.mode != 'RGB':
                im = im.convert('RGB')
            im = np.asarray(im)[:, :, ::-1]
            im = np.ascontiguousarray(im)  # contiguous
        return im

    def _single_preprocess(self, im, auto):
        """Preprocesses a single image for inference."""
        if self.transforms:
            im = self.transforms(im)  # transforms
        else:
            im = LetterBox(self.imgsz, auto=auto, stride=self.stride)(image=im)
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
        return im

    def __len__(self):
        """Returns the length of the 'im0' attribute."""
        return len(self.im0)

    def __next__(self):
        """Returns batch paths, images, processed images, None, ''."""
        if self.count == 1:  # loop only once as it's batch inference
            raise StopIteration
        auto = all(x.shape == self.im0[0].shape for x in self.im0) and self.auto
        im = [self._single_preprocess(im, auto) for im in self.im0]
        im = np.stack(im, 0) if len(im) > 1 else im[0][None]
        self.count += 1
        return self.paths, im, self.im0, None, ''

    def __iter__(self):
        """Enables iteration for class LoadPilAndNumpy."""
        self.count = 0
        return self


class LoadTensor:

    def __init__(self, imgs) -> None:
        self.im0 = imgs
        self.bs = imgs.shape[0]
        self.mode = 'image'

    def __iter__(self):
        """Returns an iterator object."""
        self.count = 0
        return self

    def __next__(self):
        """Return next item in the iterator."""
        if self.count == 1:
            raise StopIteration
        self.count += 1
        return None, self.im0, self.im0, None, ''  # self.paths, im, self.im0, None, ''

    def __len__(self):
        """Returns the batch size."""
        return self.bs


def autocast_list(source):
    """
    Merges a list of source of different types into a list of numpy arrays or PIL images
    """
    files = []
    for im in source:
        if isinstance(im, (str, Path)):  # filename or uri
            files.append(Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im))
        elif isinstance(im, (Image.Image, np.ndarray)):  # PIL or np Image
            files.append(im)
        else:
            raise TypeError(f'type {type(im).__name__} is not a supported Ultralytics prediction source type. \n'
                            f'See https://docs.ultralytics.com/modes/predict for supported source types.')

    return files


LOADERS = [LoadStreams, LoadPilAndNumpy, LoadImages, LoadScreenshots]

if __name__ == '__main__':
    img = cv2.imread(str(ROOT / 'assets/bus.jpg'))
    dataset = LoadPilAndNumpy(im0=img)
    for d in dataset:
        print(d[0])
