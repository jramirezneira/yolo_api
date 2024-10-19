# import required libraries
from vidgear.gears import NetGear, CamGear
import json
from flask import Flask, jsonify, request, Response
from flask_cors import CORS, cross_origin
import gc
from ultralytics.utils.ops import LOGGER
import threading
from utils.general import image_resize, getConfProperty, setProperty
import cv2
import torch
import traceback
from clientws4 import Custom_Stream_Class
import uvicorn
from vidgear.gears.asyncio import WebGear_RTC
from ultralytics import YOLO


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
model = YOLO("yolo11n.pt").to(device)
modelSeg = YOLO("yolo11x-seg.pt").to(device)


# video, best, cap = None, None, None


# # activate multiclient_mode mode
# options = {"multiclient_mode": True}


app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'




def cv2DestroyAllWindows():    
    cv2.destroyAllWindows()
    # setProperty("statusServer","offline")
            
    
    for obj in gc.get_objects():
        # if isinstance(obj, CamGear):
        #     try:
        #         obj.stop()
        #     except Exception as e:
        #         LOGGER.error("An exception occurred in CamGear.stop() : %s" % e)

        if isinstance(obj, Custom_Stream_Class):
            try:
                obj.source.stop()
            except Exception as e:
                LOGGER.error("An exception occurred in source.stop() : %s" % e)

            try:
                obj.source.release()
            except Exception as e:
                LOGGER.error("An exception occurred in source.release() : %s" % e)

            try:  
                obj.cv2.destroyAllWindows()                   
            except Exception as e:
                LOGGER.info("close obj.cv2.destroyAllWindows %s " % e)
    


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
    cv2DestroyAllWindows()
    url=request.args.get('url')
    type=request.args.get('type')
    response = {'message': setProperty("statusServer",'loading')}
    print("pasa 6  %s" % url) 

    instance=None
    
    # if isnew:
    try:
        for obj in gc.get_objects():
            if isinstance(obj, Custom_Stream_Class):
                instance=obj 
                break                  
        if instance is None:
            instance=Custom_Stream_Class(model, modelSeg)

        thrP = threading.Thread(target=instance.change,  args=(url, type), kwargs={})
        thrP.start() 
        # instance.change(url, type)
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
    print("pasa 7") 
    cv2DestroyAllWindows()
    response = {'message': setProperty("statusServer",'offline')}
    print("pasa 8") 
    return jsonify(response)

def service ():
    options = {"custom_stream": Custom_Stream_Class(model, modelSeg)}
    # options = {"custom_stream": Custom_Stream_Class()}

    # initialize WebGear_RTC app without any source
    # web = WebGear_RTC(source="rtsp://127.0.0.1:8554/mystream", logging=True)#, **options)
    web = WebGear_RTC( logging=True, **options)

    # run this app on Uvicorn server at address http://localhost:8080/
    uvicorn.run(web(), host="0.0.0.0", port=5000)

    # close app safely
    web.shutdown()





if __name__ == '__main__':

    # thrP = threading.Thread(target=service,  args=(), kwargs={})
    # thrP.start()  
    
    app.run(host="0.0.0.0", debug=False,  port=5001)
    
    







    
