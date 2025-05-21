# import required libraries
from utils.general import image_resize, getConfProperty, setProperty
from vidgear.gears import NetGear, CamGear
import json
from flask import Flask, jsonify, request, Response
from flask_cors import CORS, cross_origin
import gc
from ultralytics.utils.ops import LOGGER
import threading
import torch
import os
import traceback
from server.clientws3 import Custom_Stream_Class, Clientws
from vidgear.gears.asyncio import WebGear_RTC
from ultralytics import NAS, YOLO, YOLOWorld
from ultralytics import RTDETR
from ultralytics import SAM

from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import DataLoader
from torchvision import datasets
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
workers = 0 if os.name == 'nt' else 4

def collate_fn(x):
    return x[0]

dataset = datasets.ImageFolder('database')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

aligned = []
names = []
for x, y in loader:
    # print(type(x))
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        # print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])

resnet.classify = True
aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()
# [print(e1) for e1 in embeddings]
# print(embeddings)

# dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
# print(pd.DataFrame(dists, columns=names, index=names))




# video, best, cap = None, None, None


# # activate multiclient_mode mode
# options = {"multiclient_mode": True}


app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


def cv2DestroyAllWindows():        
    for obj in gc.get_objects():
        if isinstance(obj, Custom_Stream_Class):
            try:
                obj.stop()
            except Exception as e:
                LOGGER.error("An exception occurred in source.stop() : %s" % e)


setProperty("statusServer","offline")

def region_points():
    input_dict, _=getConfProperty("region_points")
    output_dict = [x for x in input_dict if x['available'] == 1]
    return output_dict

@app.route('/api/region_points', methods=['GET'])
@cross_origin()
def r_points():
    return region_points()

@app.route('/api/status', methods=['GET'])
@cross_origin()
def status():
    status, _= getConfProperty("statusServer")
    response = {'message': status}
    return jsonify(response)



@app.route('/api/start', methods=['GET'])
@cross_origin()
def start():
    try:
        # cv2DestroyAllWindows()
        sourceVideo=request.args.get('url')
        type=request.args.get('type')
        stride=int(request.args.get('stride'))
        model_input=request.args.get('model')
        response = {'message': setProperty("statusServer",'loading')}

       

        instance=None
    
        for obj in gc.get_objects():
            if isinstance(obj, Custom_Stream_Class):
                instance=obj   
                obj.restart()      
                break       

        # if instance is None:            
        #     instance=Custom_Stream_Class()
            
            

        # if instance.stride != stride:
        LOGGER.info("Cambia stride : %s" % stride)
        instance.setStride(stride)
        
        # if instance.modelName is not None:
        if instance.modelName != model_input or instance.type != type:
            print(type)
            torch.cuda.empty_cache()
            # model_input="FastSAM-s.pt"
            if type=="detection-RTDETR":
                model = RTDETR(model_input)
            elif type=="detection-YOLO-World":
                model = YOLOWorld(model_input)
            elif type=="detection-YOLO-NAS":
                model = NAS(model_input)
            elif type=="segmentation-sam2":
                model = SAM(model_input)
            elif "faceDetection":
                model = None#YOLO("best.pt")#MTCNN(image_size=160, keep_all=True, post_process=True, device=device)
            else:
                model = YOLO(model_input)
            if model is not None:
                model=model.to(device)
            LOGGER.info("Cambia type : %s" % type)
            LOGGER.info("Carga modelo : %s" % model_input)
            instance.setModelAndType(model, model_input, type)       
        

        # if instance.sourceVideo != sourceVideo:
            # if isNew == False:
            #     obj.restart()
        LOGGER.info("Cambia source : %s" % sourceVideo)
        instance.setVideo(sourceVideo)
        setProperty("statusServer",'active')

    except Exception as e:
        setProperty("statusServer","offline")
        cv2DestroyAllWindows()
        LOGGER.error("An exception occurred to open cap.release : %s" % e)
        traceback.print_exception(type(e), e, e.__traceback__)

    return jsonify(response)


@app.route('/api/stop', methods=['GET'])
@cross_origin()
def stop():    
    response = {'message': setProperty("statusServer",'offline')}
    print("pasa 7") 
    cv2DestroyAllWindows()
    
    print("pasa 8") 
    return jsonify(response)




if __name__ == '__main__':
    # thrP = threading.Thread(target=service,  args=(), kwargs={})
    # thrP.start()  
    Clientws()
    app.run(host="0.0.0.0", debug=False,  port=5002)
 





    
