import tkinter as tk
from tkinter import Label, Button, Toplevel, filedialog

from customtkinter import *

from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import os
import time
import sys

# WINDOWED MODE - disables TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

# WINDOWED MODE - logs anything in a text file that other libraries might log
log_file = open("app_log.txt", "w", encoding ="UTF-8") # Redirect stdout and stderr to a log file
sys.stdout = log_file
sys.stderr = log_file

# Set up the Tkinter window
root = CTk()
root.geometry("1280x800")
root.title("Hamster Breed Image Classification Application")
set_appearance_mode('light')

# Check if running as an executable
# Dynamic Pathing
if getattr(sys, 'frozen', False):
    # If running as a bundle (PyInstaller creates a temporary directory in sys._MEIPASS)
    base_path = sys._MEIPASS
else:
    # If running in a normal Python environment
    base_path = os.path.dirname(os.path.abspath(__file__))

# Directories
SAVE_DIR = os.path.join(base_path, "uploads/")
MODEL_DIR = os.path.join(base_path, "model/model_name.h5")

# Ensure SAVE_DIR exists
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Load the Keras model
model = tf.keras.models.load_model(MODEL_DIR)  # Replaceable with other h5 models

# Define class labels as a list
class_labels = ['Campbell', 'Roborovski', 'Syrian', 'Winter White']  # Replace with your own class names

# Global variable to store camera window reference
camera_window = None

# Function to capture and display an image from the camera
def capture_image():
    global cap
    if cap is not None:
        ret, frame = cap.read()
        if ret:
            # Resize the frame to 640x640 if necessary
            frame = cv2.resize(frame, (640, 640))

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(SAVE_DIR, f'test_image_{timestamp}.jpg')
            cv2.imwrite(save_path, frame)
            print(save_path)

            display_image(save_path)
            classify_image(save_path)
            
        else:
            print("Failed to capture image")

# Function to Upload Image  
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        display_image(file_path)
        classify_image(file_path)

# Function to display the captured image in the Tkinter window
def display_image(image_path):
    # Open and resize the image to 640x640
    image = Image.open(image_path).resize((350, 350))
    image = ImageTk.PhotoImage(image)

    display.configure(image=image)
    display.image = image

# Function to classify the captured image
def classify_image(image_path):
    try:
        image = Image.open(image_path).resize((128, 128))  # Adjust the input size as per your model's requirement
        input_data = np.array(image) / 255.0  # Normalize the image
        input_data = np.expand_dims(input_data, axis=0)

        predictions = model.predict(input_data)
        result = np.argmax(predictions)
        confidence = predictions[0][result]*100

        class_result.configure(text=f"{class_labels[result]}")
        confidence_result.configure(text=f"{confidence:.2f}%")
        
        root.update_idletasks()
        
    except Exception as e:
        # print(f"Error: {e}")  # Works in console mode
        error_dialog = tk.Toplevel(root)  # New window
        error_label = tk.Label(error_dialog, text=f"Error: {e}")
        error_label.pack()                

# Function to open a new window with the camera feed
def open_camera_window():
    global camera_window, capture_button, cap
    if camera_window is None or not camera_window.winfo_exists():
        camera_window = Toplevel(root)
        camera_window.title("Camera")

        camera_label = Label(camera_window)
        camera_label.pack()

        # Add Capture button to the camera window
        capture_button = CTkButton(camera_window, text="Capture", command=capture_image,
                                   fg_color="#007bff", hover_color="#0056b3",
                                   font=("Century Gothic", 14, "bold"), corner_radius=0,
                                   width=230, height=40)
        capture_button.pack()

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # WINDOWS ONLY
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

        def update_frame():
            global cap
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame,(640,640))
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                camera_label.imgtk = imgtk
                camera_label.configure(image=imgtk)
            if camera_window.winfo_exists():
                camera_window.after(10, update_frame)

        update_frame()
    else:
        camera_window.deiconify()

# ===================== TITLE =====================

# Hamster Image Classification Title
title = Label(root, text="Hamster Image Classification", font=("Century Gothic", 30))
title.place(relx=0.015,
            rely= 0.02,
            anchor='nw')
#title.pack()

# ===================== BUTTONS =====================

# Button to open the camera window
open_camera_button = CTkButton(root, text="Capture Image from Camera", fg_color='#5f5a5a', 
                               hover_color='#3a3b3c', command=open_camera_window, font=("Century Gothic", 14, "bold"), corner_radius=0,
                               width=230, height=40)

open_camera_button.place(relx=0.015,
                         rely= 0.1,
                         anchor='nw')

# Button to Upload Image
upload_image_button = CTkButton(root, text="Upload Image", fg_color="#007bff",
                                hover_color="#0056b3", command=upload_image, font=("Century Gothic", 18, "bold"), 
                                corner_radius=0, border_spacing=5, width=230, height=40)

upload_image_button.place(relx=0.015,
                          rely= 0.16,
                          anchor='nw')


# ===================== RESULT LABELS =====================

# Label to display the captured image
display = Label(root, borderwidth=3, relief="solid")
display.place(relx=0.32,
              rely=0.022)
# display.place(x=630, y=30)


# Labels to display the classification result
class_label = Label(root, text="Predicted Class:", font=("Century Gothic", 23))
class_label.place(relx=0.015,
                  rely=0.25)

class_result = Label(root, text=" ", font=("Century Gothic", 23, "bold"))
class_result.place(relx=0.145,
                   rely=0.25)

confidence_label = Label(root, text="Confidence:", font=("Century Gothic", 23))
confidence_label.place(relx=0.015,
                       rely=0.30)

confidence_result = Label(root, text=" ", font=("Century Gothic", 23, "bold"))
confidence_result.place(relx=0.12,
                        rely=0.30)

# Run the Tkinter main loop
root.mainloop()

log_file.close()