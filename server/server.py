# import required libraries
from vidgear.gears import NetGear, CamGear
import json
from flask import Flask, jsonify, request, Response
from flask_cors import CORS, cross_origin
import gc
from ultralytics.utils.ops import LOGGER
import threading
from ultralytics import YOLO
from utils.stream_loaders import LoadImages, LoadStreamNoThread
import time
from utils.general import image_resize, getConfProperty, setProperty
import cv2
import boto3
import torch
import subprocess
from subprocess import Popen
import os
import socket


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
model = YOLO("yolov8n.pt").to(device)


video, best, cap = None, None, None


# activate multiclient_mode mode
options = {"multiclient_mode": True}


app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'



server = NetGear(
    address="0.0.0.0",
    port=["5567"],
    protocol="tcp",
    pattern=2,
    logging=True,
    **options
)



def cv2DestroyAllWindows():    
    cv2.destroyAllWindows()
    setProperty("statusServer","offline")
            
    for obj in gc.get_objects():
        if isinstance(obj, LoadStreamNoThread):
            try:
                if obj.cap.stop:
                    obj.cap.stop() 
                    LOGGER.info("obj.cap.stop() %s " % obj)
            except Exception as e:
                LOGGER.error("An exception occurred in obj.cap.stop : %s" % e)
            try:
                if obj.cap.release:
                    obj.cap.release() 
                    LOGGER.info("obj.cap.release() %s " % obj)
            except Exception as e:
                LOGGER.error("An exception occurred in obj.cap.release : %s" % e)
            try:  
                obj.cv2.destroyAllWindows()   
                LOGGER.info("close obj.cv2.destroyAllWindows %s " % obj)
            except Exception as e:
                LOGGER.error("An exception occurred in obj.cv2.destroyAllWindows : %s" % e)
            try: 
                obj.thrP.join()
                LOGGER.info("close obj.thrP.join %s " % obj)
            except Exception as e:
                LOGGER.error("An exception occurred in obj.thrP.join : %s" % e)


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
    response = {'message': getConfProperty("statusServer")}
    return jsonify(response)



@app.route('/api/start', methods=['GET'])
@cross_origin()
def start():
    cv2DestroyAllWindows()
    url=request.args.get('url')
    response = {'message': setProperty("statusServer",'active')}
    print("pasa 6  %s" % url) 

    try:
        ldst = LoadStreamNoThread(url)
        ldst.startPrediction(server)
    except Exception as e:
        setProperty("statusServer","offline")
        cv2DestroyAllWindows()
        LOGGER.error("An exception occurred to open cap.release : %s" % e)
    return jsonify(response)


@app.route('/api/stop', methods=['GET'])
@cross_origin()
def stop():    
    print("pasa 7") 
    cv2DestroyAllWindows()
    response = {'message': setProperty("statusServer",'offline')}
    print("pasa 8") 
    return jsonify(response)




if __name__ == '__main__':
    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(dir_path)
        proc = subprocess.Popen (['python',  '%s/clientws.py' % dir_path]) 
    except Exception as e:
        LOGGER.warning("Error subprocess.Popen : %s" % e)
    app.run(host="0.0.0.0", debug=False,  port=5001)







    
