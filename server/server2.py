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
from ultralytics import YOLO
from ultralytics import RTDETR

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')





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
           
            # try:  
            #     obj.cv2.destroyAllWindows()                   
            # except Exception as e:
            #     LOGGER.info("close obj.cv2.destroyAllWindows %s " % e)
            
            # try:  
            #     obj.thrP.join()                   
            # except Exception as e:
            #     LOGGER.info("close obj.thrP.join() %s " % e)

            # try:
            #     del obj
            # except Exception as e:
            #     LOGGER.error("An exception occurred in del obj : %s" % e)

       
        # if isinstance(obj, Clientws):
        #     try:               
        #         obj.shutdown()
        #     except Exception as e:
        #         LOGGER.error("An exception occurred in del obj : %s" % e)


            
    


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
            # model_input="FastSAM-s.pt"
            if type=="detection-RTDETR":
                model = RTDETR(model_input).to(device)
            else:
                model = YOLO(model_input).to(device)
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





# def service ():
   
#     options = {
#         "custom_stream": Custom_Stream_Class(),
#     # "frame_size_reduction": 5,
#       "enable_live_broadcast": True,
#       "enable_infinite_frames":True,
#       "CAP_PROP_FPS":20
#     #   "CAP_PROP_FRAME_WIDTH":320, "CAP_PROP_FRAME_HEIGHT":240, "CAP_PROP_FPS":60
# }
#     # options = {"custom_stream": Custom_Stream_Class()}

#     # initialize WebGear_RTC app without any source
#     # web = WebGear_RTC(source="rtsp://127.0.0.1:8554/mystream", logging=True)#, **options)
#     web = WebGear_RTC( logging=True, **options)

#     # run this app on Uvicorn server at address http://localhost:8080/
#     uvicorn.run(web(), host="0.0.0.0", port=5003)
    
#     # close app safely
#     web.shutdown()




if __name__ == '__main__':
    # thrP = threading.Thread(target=service,  args=(), kwargs={})
    # thrP.start()  
    Clientws()
    app.run(host="0.0.0.0", debug=False,  port=5002)
 





    
