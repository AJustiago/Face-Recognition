from flask import Flask, Response, request, jsonify
from flask_cors import CORS, cross_origin
import cv2
import pickle
from keras.models import load_model
from keras.utils import img_to_array
from PIL import Image as im
import numpy as np
import base64
import os
import io
from deepface import DeepFace
import psycopg2
import glob
import sklearn
from deepface.commons import functions
from deepface.basemodels import Facenet, ArcFace
import time
import uuid
from datetime import datetime  
import json
from dotenv import load_dotenv
import os 
from gevent import socket
from gevent.pywsgi import WSGIServer

socket.socket = socket.socket

app = Flask(__name__)
CORS(app)

load_dotenv('./.env')

# Taking Value from .env
db_host = os.getenv("db_host")
db_database = os.getenv("db_database")
db_user = os.getenv("db_user")
db_password = os.getenv("db_password")

file_temp = os.getenv("path_upload")
file_db = os.getenv("path_localdb")

# Liveness Detection & Mask Detection
protoPath = './face_detector/deploy.prototxt'
modelPath = './face_detector/res10_300x300_ssd_iter_140000.caffemodel'

try:
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
except:
    response_data = {"success": False,
                     "data": "-",
                     "message": "error connecting model",
                     "error": 500}
modelLive = load_model('./liveness/liveness.model')
try:
    leLive = pickle.loads(open('./liveness/le.pickle', "rb").read())
except:
    response_data = {"success": False,
                     "data": "-",
                     "message": "error connecting model",
                     "error": 500}
    
modelMask = load_model('./mask/MaskDetection.model')
try:
    leMask = pickle.loads(open('./mask/md.pickle', "rb").read())
except:
    response_data = {"success": False,
                     "data": "-",
                     "message": "error connecting model",
                     "error": 500}

# Database Access
def create_connection():
    try:
        conn = psycopg2.connect(host = db_host,database = db_database ,user = db_user ,password = db_password)
        cur = conn.cursor()
        return conn, cur
    except:
        return False

def Collect_NameDB(id_result):
    try:
        conn, cur = create_connection()
        try: 
            statement = "SELECT employee_name FROM employee WHERE no_ktp = %(no_ktp)s"
            cur.execute(statement, {'no_ktp': id_result})
            id_result = cur.fetchall()
        except(Exception, psycopg2.Error) as error:
            return False
        finally:
            conn.commit()
            conn.close()
    finally:
        return id_result

# Pickle Update
def build_model(model_name):

    try:# singleton design pattern
        global model_obj

        models = {
            "ArcFace": ArcFace.loadModel,
        }

        if not "model_obj" in globals():
            model_obj = {}

        if not model_name in model_obj:
            model = models.get(model_name)
            if model:
                model = model()
                model_obj[model_name] = model
            else:
                raise ValueError(f"Invalid model_name passed - {model_name}")

        return model_obj[model_name]
    except(Exception) as error:
        return False

def represent(
    img_path,
    model_name="ArcFace",
    enforce_detection=False,
    detector_backend="opencv",
    align=True,
    normalization="base",
):
    try:
        resp_objs = []
        if build_model == False:
            return False
        model = build_model(model_name)

        target_size = functions.find_target_size(model_name=model_name)
        if detector_backend != "skip":
            img_objs = functions.extract_faces(
                img=img_path,
                target_size=target_size,
                detector_backend=detector_backend,
                grayscale=False,
                enforce_detection=enforce_detection,
                align=align,
            )
        else:  
            if isinstance(img_path, str):
                img = functions.load_image(img_path)
            elif type(img_path).__module__ == np.__name__:
                img = img_path.copy()
            else:
                raise ValueError(f"unexpected type for img_path - {type(img_path)}")

            if len(img.shape) == 4:
                img = img[0]  # e.g. (1, 224, 224, 3) to (224, 224, 3)
            if len(img.shape) == 3:
                img = cv2.resize(img, target_size)
                img = np.expand_dims(img, axis=0)

            img_region = [0, 0, img.shape[1], img.shape[0]]
            img_objs = [(img, img_region, 0)]


        for img, region, confidence in img_objs:
            # custom normalization
            img = functions.normalize_input(img=img, normalization=normalization)

            # represent
            if "keras" in str(type(model)):
                # new tf versions show progress bar and it is annoying
                embedding = model.predict(img, verbose=0)[0].tolist()
            else:
                # SFace and Dlib are not keras models and no verbose arguments
                embedding = model.predict(img)[0].tolist()

            resp_obj = {}
            resp_obj["embedding"] = embedding
            resp_obj["facial_area"] = region
            resp_obj["face_confidence"] = confidence
            resp_objs.append(resp_obj)

        return resp_objs
    except(Exception) as error:
        return False

# cors = CORS(app, resources={r"/api/capture": {"origins": "*"}})
# app.config['CORS_HEADERS'] = 'Content-Type'

#---------------------------------#
#Capturing Image from Video Stream#
#---------------------------------#
@app.route('/api/capture', methods=['POST','GET'])
@cross_origin()
def receive_string():
    response_data = ' '
    label = ' '
    try:
        data = request.json['image']
        data_split = data.split(",")
    except(Exception) as error:
        response_data = {"success": False,
                         "data" : "-",
                         "message" : "invalid data",
                         error : 405}
        return jsonify(response_data)
    try:
        image_64 = data_split[1]
    except:
        response_data = {"success": False,
                         "data" : "-",
                         "message" : "invalid data",
                         error : 405}
        return jsonify(response_data)
    image_bytes=base64.b64decode((image_64))
    image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    temp_uuid = str(uuid.uuid4())
    try:    
        file_path = file_temp + temp_uuid + '_img.jpeg'
    except(Exception) as errror:
        response_data = {"success": False,
                         "data": "-",
                         "message": "environtment not found",
                         error : 500}
    image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image = im.fromarray(image)
    # image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    # image = im.fromarray(image)
    # image = image.resize((224,224))
    # file_path = 'TesPython/backend/Database/database/img.jpeg'
    # image.save(file_path) 
    # ioimage = io.BytesIO(image)
    # image_array = im.open(ioimage)
    blob = cv2.dnn.blobFromImage(cv2.resize(image_array, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    (h, w) = (600, 600)
    # try:
    detections = net.forward()
    # except:
    #     response_data = {"success": False,
    #                     "data": "-",
    #                     "message": "Face Not Detected",
    #                     "error": "-"}
        # return jsonify(response_data)
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            face = image_array[startY:endY, startX:endX]
            face = cv2.resize(face, (32, 32))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
            preds = modelLive.predict(face)[0]
            j = np.argmax(preds)
            label1 = leLive.classes_[j]
            if "real" in label1:
                preds = modelMask.predict(face)[0]
                j = np.argmax(preds)
                label2 = leMask.classes_[j]
                if "without_mask" in label2:
                    data_img = {'UUID' : temp_uuid}
                    image.save(file_path) 
                    response_data = {"success": True,
                                    "data": temp_uuid,
                                    "message": "Face Detected",
                                    "error": "200"}
                    response = jsonify(response_data)
                    return response
                else:
                    response_data = {"success": False,
                                    "data": "-",
                                    "message": "Mask Detected",
                                    "error": "-"}
                    response = jsonify(response_data)
                    return response
            else:
                response_data = {"success": False,
                                "data": "-",
                                "message": "Face Not Detected",
                                "error": "-"}
                response = jsonify(response_data)
                return response

#---------------------------------#
#Check Data Capture with database #
#---------------------------------#
@app.route('/api/facedb', methods=['POST'])
def facerecognition():
    try:
        lenjpegs = len(glob.glob('./Database/database/*'))
        # file_path = './upload/img.jpeg'
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        try:
            data = request.json['ImgUUID']
        except(Exception) as error:
            response_data = {"success": False,
                             "data": "-",
                             "message": "invaid data",
                             "error": 405}
            response = jsonify(response_data)
            return response
        try:
            file_path = file_temp + data +'_img.jpeg'
        except (Exception) as error:
            response_data = {"success": False,
                             "data" : "-",
                             "message": "env not found",
                             "error": 500}
            response = jsonify(response_data)
            return response
        
        # data_split = data.split(",")
        # image_64 = data_split[1]
        # image_bytes=base64.b64decode((image_64))
        # image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
        # image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        # image = im.fromarray(image)
        # image.save(file_path) 

        if lenjpegs != 0:        
            dff = DeepFace.find(img_path = file_path, db_path = file_db, model_name ="ArcFace", distance_metric= "cosine", enforce_detection=False)
            if len(dff) > 1:
                if len(dff[0]) > len(dff[1]):
                    dff = dff[1]
                else:
                    dff = dff[0]
            else:
                dff = dff[0]
            if dff.shape[0] >= 1:
                matched = dff.iloc[0].identity
                if type(matched) == list:
                    matched = matched[0]
                dfv = DeepFace.verify(img1_path=file_path, img2_path=matched, model_name="ArcFace", distance_metric= "cosine", enforce_detection=False)

                if dfv["verified"] == True:
                    Id_Result = os.path.basename(matched).split('/')[0]
                    Id_Result = os.path.splitext(Id_Result)[0]
                    if Collect_NameDB(Id_Result) == False:
                        response_data = {"success": False,
                            "data": "-",
                            "message": "error selecting data from database",
                            "error": 500}
                        response = jsonify(response_data)
                        return response
                    Name_DB = Collect_NameDB(Id_Result)
                    if len(Name_DB) <= 0:
                        response_data = {"success": False,
                                        "data": "-",
                                        "message": "data not found in database",
                                        "error": "200"}
                        response = jsonify(response_data)
                        return response
                    Name_Result = (Name_DB[0])[0]

                    if os.path.exists(file_path):
                        os.remove(file_path)

                    data_db = {"Id": Id_Result,
                               "Name": Name_Result}
                    response_data = {"success": True,
                                    "data": json.dumps(data_db),
                                    "message": "data found",
                                    "error": "200"}
                    # if os.path.exists(file_path):
                    #     os.remove(file_path)
                    response = jsonify(response_data)
                    return response

        response_data = {"success": False,
                        "data": "-"}
        response = jsonify(response_data)
        return response
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        response_data = {"success": False,
                        "data": "-",
                        "message": "error occured",
                        "error": str(e)}
        response = jsonify(response_data)
        return response
    
#---------------------------------#
#Show Result Database             #
#---------------------------------#
@app.route('/api/formdb', methods=['POST'])
def formdb():
    response_data =' '
    try:
        data = request.json
        id = data.get('id')
        name = data.get('name')
        ktp = data.get('no_ktp')
        image_UUID = data.get('imageUUID')

        # Image Save
        try:
            img_path = file_temp + image_UUID +'_img.jpeg'
        except(Exception) as error:
            response_data = {"success": False,
                         "data": "-",
                         "message": "environtment not found",
                         error : 500}
            response = jsonify(response_data)
            return response
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = im.fromarray(image)
        # image_64 = data_split[1]
        # image_bytes=base64.b64decode((image_64))
        # image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
        # image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        # image = im.fromarray(image)
        try:
            file_path = file_db + '/database/' + ktp + '.jpeg'
        except(Exception) as error:
            response_data = {"success": False,
                         "data": "-",
                         "message": "environtment not found",
                         error : 500}
            response = jsonify(response_data)
            return response
        image.save(file_path)
        
        # Created_Date
        timestamp = time.time()
        print("Timestamp:", timestamp)
        date_time = datetime.fromtimestamp(timestamp)
        created_date = date_time.strftime("%Y-%m-%d %H:%M:%S")

        # Defunct_ind
        defunct_ind = False

        #Save ID, NAME, Created_Date, Defunct_id to Database
        if create_connection() == False:
            response_data = {"success": False,
                            "data": "-",
                            "message": "error connecting database",
                            error : 500}
            response = jsonify(response_data)
            return response
        conn, cur = create_connection()
        try: 
            statement = "INSERT INTO Employee (employee_id, employee_name, no_ktp, created_date, defunct_ind) VALUES (%s, %s, %i, %s, %s)"
            cur.execute(statement, (id, name, ktp, created_date, defunct_ind))
        except(Exception, psycopg2.Error) as error:
            print(psycopg2.Error)
        finally:
            conn.commit()
            conn.close()

        if os.path.exists(file_db + "/representations_arcface.pkl"):

            #Append Pickle
            employee = [] 
            
            db_path = "./Database"
            img_path = file_path
            model_name = "ArcFace"
            align = True
            
            file_name = "representations_temp.pkl"
            file_name = file_name.replace("-", "_").lower()
            
            target_size = functions.find_target_size(model_name="ArcFace")
            
            employee.append(img_path)
            
            if len(employee) == 0:
                response_data = {"success": False,
                            "data": "-",
                            "message": "There is no image, Validate .jpg or .png files exist in this path.",
                            "error": 500}
                response = jsonify(response_data)
                return response

            representations = []
            img_objs = functions.extract_faces(
                img=employee[0],
                target_size=target_size,
                detector_backend="opencv",
                grayscale=False,
                align=True,
                enforce_detection=False
                )
            
            if represent(img_path=img_objs[0][0],
                        model_name=model_name,
                        detector_backend="skip",
                        align=align,
                        normalization="base",
                        enforce_detection=False
                        ) == False:
                response_data = {"success": False,
                         "data": "-",
                         "message": "error building model pickle",
                         "error": 500}
                response = jsonify(response_data)
                return response
            else: embedding_obj = represent(
                                img_path=img_objs[0][0],
                                model_name=model_name,
                                detector_backend="skip",
                                align=align,
                                normalization="base",
                                enforce_detection=False
                                )
            img_representation = embedding_obj[0]["embedding"]
            
            instance = []
            instance.append(employee)
            instance.append(img_representation)
            representations.append(instance)
            
            with open(f"{db_path}/{file_name}", "wb") as f:
                pickle.dump(representations, f)
            
            print( f"Representations stored in {db_path}/{file_name} file.")
            folder= file_db + '/'
            db = {}
            try:
                for filename in os.listdir(folder):  
                    if filename.endswith('.pkl'):
                        myfile = open(folder+filename,"rb")
                        db[os.path.splitext(filename)[0]]= pickle.load(myfile)
                        myfile.close()
                        print(filename)
            except:
                response_data ={"success": False,
                                "data": "-",
                                "message": "file .pkl not found",
                                "error": 500}
            try:
                for i in db["representations_temp"]:
                        db["representations_arcface"].append(i)
            except:
                response_data ={"success": False,
                                "data": "-",
                                "message": "model representations not found",
                                "error": 500}
            try:
                file = open(folder + "representations_arcface.pkl","wb")
                pickle.dump(db["representations_arcface"], file)
                file.close()
            except:
                response_data ={"success": False,
                                "data": "-",
                                "message": "file .pkl not found",
                                "error": 500}
            
            if os.path.exists(folder + "representations_temp.pkl"):
                os.remove(folder + "representations_temp.pkl")
            response_data = {"success": True,
                            "data": "-",
                            "message": "data uploaded",
                            "error": "200"}
    
    except Exception as e:
        response_data = {"success": False,
                        "data": "-",
                        "message": "error occured",
                        "error": str(e)}
        return jsonify(response_data)
    # if os.path.exists(file_path):
    #     os.remove(file_path)
    response_data = {"success": True,
                    "data": "-",
                    "message": "first data uploaded",
                    "error": "200"}
    return jsonify(response_data)

#---------------------------------#
#       Testing Network           #
#---------------------------------#
@app.route('/')
@cross_origin()
def start():
    return jsonify({"status": 200, 
                    "data": "-", 
                    "message": "OK", 
                    "success": "success" }), 200

if __name__ == '__main__':
    try:
        http_server = WSGIServer(("0.0.0.0", 7375), app)
        http_server.serve_forever()
    except SystemExit:
        pass

