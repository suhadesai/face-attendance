from flask import Flask, render_template, Response, send_file, url_for, redirect, request
import os
import cv2
import joblib
import csv
import time
import numpy as np
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv



app = Flask(__name__)
camera = None

load_dotenv
MONGODB_URI = os.environ.get("MONGODB_URI")
#MongoDB connection
try:
   client = MongoClient(MONGODB_URI)
   db = client['face-recogntion']
   employees_collection = db['employees']
   attendance_collection = db['attendance']
   print("Successfully connected to MongoDB!")
except Exception as e:
   print(f"Could not connect to MongoDB: {e}")


DISTANCE_THRESHOLD = 0.9
MODEL_PATH = 'models/model_and_classifier.joblib'
EMPLOYEE_DIR = 'employees'
CONFIRMATION_TIME = 2
ATTENDANCE_FILE = 'attendance.csv'




capture = None
last_detected_ID = None
detection_timer = None
changed_mode_timer = None
modeType = 2


classifier = None
labelEncoder = None
faceDetector = FaceNet()
nnModel = None
y = []


def load_model_and_classifier():
   global classifier, labelEncoder, nnModel, y


   try:
       classifier, labelEncoder, nnModel, y = joblib.load(MODEL_PATH)
       return True
   except Exception as e:
       if train_model():
           classifier, labelEncoder, nnModel, y = joblib.load(MODEL_PATH)
           return True
       else:
           print("failed to load or train model.")
           return False


def train_model():
   global classifier, labelEncoder, nnModel, y


   X = []
   y = []


   for employee in os.listdir(EMPLOYEE_DIR):
       employeePath = os.path.join(EMPLOYEE_DIR, employee)


       if not os.path.isdir(employeePath):
           print(f"Skipping {employeePath} as it is not a directory")
           continue


       for imageNum in os.listdir(employeePath):
           imagePath = os.path.join(employeePath, imageNum)


           if not (imageNum.lower().endswith(('.jpg', '.jpeg', '.png'))):
               print(f"Skipping non-image file: {imagePath}")
               continue


           imageRead = cv2.imread(imagePath)
           image = cv2.cvtColor(imageRead, cv2.COLOR_BGR2RGB)


           detected_faces = faceDetector.extract(image, threshold=DISTANCE_THRESHOLD)
           if not detected_faces:
               continue


           x_coord, y_coord, w, h = detected_faces[0]['box']


           x_min = max(0, x_coord)
           y_min = max(0, y_coord)
           x_max = min(image.shape[1], x_coord + w)
           y_max = min(image.shape[0], y_coord + h)


           cropped_face = image[y_min:y_max, x_min:x_max]


           embeddings = faceDetector.embeddings([cropped_face])
           if embeddings is not None and len(embeddings) > 0:
               X.append(embeddings[0])
               y.append(employee)


   if len(X) == 0:
       print("No images detected for training.")
       return False


   X = np.array(X)
   y = np.array(y)


   labelEncoder = LabelEncoder()
   encoded_y = labelEncoder.fit_transform(y)


   classifier = KNeighborsClassifier(n_neighbors=1)
   classifier.fit(X, encoded_y)


   nnModel = NearestNeighbors(n_neighbors=1)
   nnModel.fit(X)


   os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
   joblib.dump((classifier, labelEncoder, nnModel, encoded_y), MODEL_PATH)
   print("Model trained and saved successfully.")
   return True


def mark_attendance(employee_id):
   date = datetime.now().strftime("%D")
   time = datetime.now().strftime("%H:%M")


   existing_record = attendance_collection.find_one({
       'employee_id': employee_id,
       'date': date
   })
  
   if existing_record:
       print(f"Attendance for {employee_id} already marked today.")
       return False
   else:


       employee_info = employees_collection.find_one({'id': employee_id})
       employee_name = employee_info['name'] if employee_info else 'Unknown'
       
       attendance_record = {
           'employee_id': employee_id,
           'name': employee_name,
           'date': date,
           'time': time
       }
       attendance_collection.insert_one(attendance_record)
          
       with open(ATTENDANCE_FILE, 'a', newline='') as f:
           writer = csv.writer(f)
           if not os.path.exists(ATTENDANCE_FILE):
               writer.writerow(["Employee ID", "Name", "Date", "Time"])
           writer.writerow([employee_id, employee_name, date, time])
  
       print(f"Marked attendance for {employee_id} ({employee_name}) at {time} in both DB and CSV.")
       return True


def generate_frames():
   global camera, modeType, background, last_detected_ID, detection_timer, changed_mode_timer
   camera = cv2.VideoCapture(0) # change so that it displays 1 if 0 fails
   if not camera.isOpened():
       camera = cv2.VideoCapture(1)
       print("Error: Could not open camera.")
       return


   if not load_model_and_classifier():
       print("Failed to load model. Exiting stream.")
       return


   while True:
       success, frame = camera.read()
       if not success:
           break


       frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       faces = faceDetector.extract(frame_rgb, threshold=DISTANCE_THRESHOLD)


       if len(faces) == 0:
           modeType = 2
           last_detected_ID = None
           detection_timer = None
           changed_mode_timer = None
       else:
           for face in faces:
               x, box_y, w, h = face['box']
               if w <= 0 or h <= 0:
                   continue
               x_min = max(0, x)
               y_min = max(0, box_y)
               x_max = min(frame.shape[1], x + w)
               y_max = min(frame.shape[0], box_y + h)


               face_image = frame_rgb[y_min:y_max, x_min:x_max]


               if face_image.size == 0:
                   continue


               embeddings_list = faceDetector.embeddings([face_image])
               if len(embeddings_list) == 0:
                   continue


               embedding = embeddings_list[0]


               probs = classifier.predict_proba([embedding])[0]
               pred = np.argmax(probs)
               employeeID_pred = labelEncoder.inverse_transform([pred])[0]
               distances = nnModel.kneighbors([embedding])
               distance = float(distances[0][0])


               if distance < DISTANCE_THRESHOLD:
                   employee_ID = employeeID_pred
                   employee_info = employees_collection.find_one({'id': employee_ID})
                   if employee_info:
                       employee_name = employee_info['name']
                   else:
                       employee_name = "Unknown"
                   color = (0, 255, 0)  # green box
                   if last_detected_ID == employee_ID:
                       if detection_timer is not None and time.time() - detection_timer >= CONFIRMATION_TIME:
                           if changed_mode_timer is not None and time.time() - changed_mode_timer < CONFIRMATION_TIME:
                               modeType = 1
                           elif changed_mode_timer is not None and time.time() - changed_mode_timer >= CONFIRMATION_TIME:
                               modeType = 0
                               changed_mode_timer = None
                           elif mark_attendance(employee_ID):
                               modeType = 1
                               changed_mode_timer = time.time()
                           else:
                               modeType = 0
                       else:
                           modeType = 2
                           if detection_timer is None:
                               detection_timer = time.time()
                   else:
                       modeType = 2
                       last_detected_ID = employee_ID
                       detection_timer = time.time()


               else:
                   employee_ID = "Unknown"
                   employee_name = "Unknown"
                   color = (0, 0, 255)  # red box
                   last_detected_ID = None
                   detection_timer = None
                   changed_mode_timer = None
                   modeType = 2


               cv2.rectangle(frame, (x, box_y), (x + w, box_y + h), color, 2)
               cv2.putText(frame, f'{employee_name} D:{distance:.2f}', (x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


       ret, buffer = cv2.imencode('.jpg', frame)
       frame = buffer.tobytes()
       yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




@app.route('/')
def index():
   global camera
   camera = None
   return render_template('index2.html')


@app.route('/markingfortoday')
def mark_attendance_route():
   global camera
   camera = cv2.VideoCapture(0)
   if not camera.isOpened():
       camera = cv2.VideoCapture(1)
   return render_template('index.html')


@app.route('/register', methods = ['GET', 'POST'])
def register():
   global camera
   camera = None
   if request.method == 'POST':
       employee_name = request.form.get('employee_name')
       employee_id = request.form.get('employee_id')
       images = request.files.getlist('images')


       existing_employee = employees_collection.find_one({'id': employee_id})
      
       if existing_employee:
           return f"Error: An employee with ID '{employee_id}' already exists. Please use a unique ID."
          
       if not images or images[0].filename == '':
           return "Error: Please upload at least one image."


       employee_folder = os.path.join(EMPLOYEE_DIR, employee_id)
       os.makedirs(employee_folder, exist_ok=True)
      
       num_images = 0
       for image in images:
           image.save(os.path.join(employee_folder, image.filename))
           num_images += 1


       employee_data = {
           'id': employee_id,
           'name': employee_name,
           'image_folder': employee_folder,
           'num_images': num_images
       }
       employees_collection.insert_one(employee_data)
      
      
       # re-train the model with the new employee's data
       train_model()


       return redirect(url_for('registration_success'))
  
   return render_template('register.html')




@app.route('/attendance')
def attendance():
   # global camera
   # camera = None
   data = []
   header = ["Employee ID", "Name", "Date", "Time"]
   try:
       with open('attendance.csv', 'r') as file:
           csv_reader = csv.reader(file)
           for row in csv_reader:
               data.append(row)
   except FileNotFoundError:
       header = ["Error"]
       data.append(["Attendance file not found."])
   return render_template('attendance.html', header = header, data = data)


@app.route('/registration-success')
def registration_success():
   # global camera
   # camera = None
   return render_template('registration-success.html')


@app.route('/video')
def video():
   return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/changedMode')
def changed_mode():
   global modeType
   return send_file(f'static/mode_{modeType}.JPG', mimetype='image/JPG')




if __name__ == '__main__':
   app.run(debug=True)
