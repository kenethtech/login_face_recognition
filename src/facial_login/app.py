"""
Enables users to login with their faces or login details
"""

import toga
from toga.style.pack import COLUMN, ROW
from toga.style import Pack
import threading
import cv2
import face_recognition
import firebase_admin
from firebase_admin import credentials, firestore
import bcrypt
import os
import numpy as np



base_dir = os.path.dirname(os.path.abspath(__file__))
firebase_key = os.path.join(base_dir, "firebase_key.json")
#-----Firebase setup--------#
cred = credentials.Certificate(firebase_key)
firebase_admin.initialize_app(cred)
db = firestore.client()

#----- capture face help method
def capture_face_embeddings(self):
    cap = cv2.VideoCapture(0)
    embedding = None
    ret, frame = cap.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            embedding = face_encodings[0].tolist() #Take first face found
    
    cap.release()
    cv2.destroyAllWindows()
    return embedding

#-----Hashed password ----
def hashed_password(password):
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt)

def check_password(password, password_hash):
    return bcrypt.checkpw(password.encode(), password_hash.encode())

#------Save user ----
def save_user(username, email, password_hash, embedding):
    db.collection('users').document(username).set({
        "email": email,
        "password": password_hash,
        "face_embedding": embedding
    })

def get_user(email):
    doc = db.collection("users").document(email).get()

    if doc.exists:
        return doc.to_dict()
    
    return None

#-----Method to compare the live face embedding and the stored face embedding using cosine similarity----
def compare_embeddings(live_embedding, stored_embedding, threshold=0.6):
    face_embedding_one = np.array(live_embedding)
    face_embedding_two = np.array(stored_embedding)

    similarity = np.dot(face_embedding_one, face_embedding_two)/ (np.linalg.norm(face_embedding_one) * np.linalg.norm(face_embedding_two))
    return similarity >= threshold


#----Main Application code ----
class Faciallogin(toga.App):
    def startup(self):
       self.main_window = toga.MainWindow(title= "Face login system")

       self.login_view = self.build_login_screen()
       self.register_view = self.build_register_screen()
       self.main_window.content = self.login_view
       self.main_window.show()


    #----Login Screen----
    def bulid_login_screen(self):
        self.log_email = toga.TextInput(placeholder="Email")
        self.log_password = toga.PasswordInput(placeholder="password")

        self.face_embeddings = None

        log_btn = toga.Button("Login with Email/Password", on_press= self.login_password_user)
        face_login_btn = toga.Button("Login with face", on_press= self.login_face_user)
        register_btn = toga.Button("Don't have an account? Register", on_press= self.show_registration)

        #------status label----
        self.status_label = toga.Label("")

        box = toga.Box(style= Pack(direction=COLUMN, padding=20))
        box.add(self.log_email)
        box.add(self.log_password)
        box.add(log_btn)
        box.add(face_login_btn)
        box.add(register_btn)
        box.add(self.status_label)

        return box
    
    #-----Registration screen ----
    def build_register_screen(self):
        self.reg_username = toga.TextInput(placeholder="Username")
        self.reg_email = toga.TextInput(placeholder="Email")
        self.reg_password = toga.PasswordInput(placeholder="PassWord")

        self.face_embeddings = None

        capture_btn = toga.Button("Capture Face", on_press= self.capture_face)
        register_button = toga.Button("Register", on_press= self.register_user)
        back_btn = toga.Button("Back", on_press=self.show_login)

        #------Status label----
        self.status_label = toga.Label("")

        box = toga.Box(style= Pack(direction=COLUMN, padding=20))
        box.add(self.reg_username)
        box.add(self.reg_email)
        box.add(self.reg_password)
        box.add(capture_btn)
        box.add(register_button)
        box.add(back_btn)
        box.add(self.status_label)

        return box
    
    #--------Navigation ----
    def show_registration(self, widget):
        self.main_window.content = self.register_view
    
    def show_login(self, widget):
        self.main_window.content = self.login_view

    #---- face capturing action ---
    def capture_face(self, widget):
        def run_capture():
            self.status_label.text = "Opening the Webcame....."
            embedding = capture_face_embeddings()
            if embedding:
                self.face_embeddings = embedding
                self.status_label.text = "Face Captured Successful!"

            else:
                self.status_label.text = " No face detected!!"
        
        threading.Thread(target=run_capture).start()
    

    #---- register the user---
    def register_user(self, widget):
        username = self.reg_username.value
        email = self.reg_email.value
        password = self.reg_password.value
        embedding = self.face_embeddings

        if not username or not email or not password or not embedding:
            self.status_label.text = "All fields and face capture are required!"
            return
        
        hashed_pw = hashed_password(password)
        save_user(email, hashed_pw, embedding)
        self.status_label.text = f"User {email} registered successfuly!"

        
        #----allow users to login with password and email
    def login_password_user(self, widget):
        email = self.log_email.value
        password = self.log_password.value

        if not email or not password:
            self.status_label.text = "Enter your Email and password"
            return
        
        user = get_user(email)
        if user and check_password(password,user["password_hash"]):
            self.status_label.text = f"Welcome {email}!"
            #----After login logic -----

        else:
            self.status_label.text = "Invalid Email or Password"
        
    #----Face login code ---
    def login_face_user(self, widget):
        def run_face_login():
            self.status_label.text = "opening Webcam..."
            live_embedding = capture_face_embeddings()

            if not live_embedding:
                self.status_label.text = "No face detected!"
                return
            
            users = db.collection("users").get()

            for user in users:
                data = user.to_dict()

                if compare_embeddings(live_embedding, data["face_embedding"]):
                    self.status_label.text = f" Face login successful, welcome {user.id}!"
                    return
                else:
                    self.status_label.text = " Face Login failed! Try again later!"

            
        threading.Thread(target=run_face_login).start()



def main():
    return Faciallogin()
