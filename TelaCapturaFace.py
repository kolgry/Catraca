import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import re
import datetime


# A basic resize_video function, replace with your actual helper_functions.py if different
def resize_video(width, height, max_width = 600):
  # max_width = in pixels. define the maximum width of the processed video.
  # the height will be proportional (defined in the calculations below)

  # if resize=True the saved video will have his size reduced ONLY IF its width is bigger than max_width
  if (width > max_width):
    # we need to make width and height proportionals (to keep the proportion of the original video) so the image doesn't look stretched
    proportion = width / height
    # to do it we need to calculate the proportion (width/height) and we'll use this value to calculate the new height
    video_width = max_width
    video_height = int(video_width / proportion)
  else:
    video_width = width
    video_height = height

  return video_width, video_height


# Function to parse the name of the person, which will the name of the subdirectory
def parse_name(name):
    name = re.sub(r"[^\w\s]", '', name)  # Remove all non-word characters (everything except numbers and letters)
    name = re.sub(r"\s+", '_', name)  # Replace all runs of whitespace with a single underscore
    return name


# Create the final folder where the photos will be saved (if the path already doesn't exist)
def create_folders(final_path, final_path_full):
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    if not os.path.exists(final_path_full):
        os.makedirs(final_path_full)


# Return the detected face using Haar cascades
def detect_face(face_detector, orig_frame):
    frame = orig_frame.copy()  # to keep the original frame intact
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 5)

    face_roi = None
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        face_roi = orig_frame[y:y + h, x:x + w]  # get ROI (region of interest) of the face
        face_roi = cv2.resize(face_roi, (140, 140))  # resize the face to a fixed size.
    return face_roi, frame


# Return the detected face using SSD
def detect_face_ssd(network, orig_frame, show_conf=True, conf_min=0.7):
    frame = orig_frame.copy()  # to keep the original frame intact
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    network.setInput(blob)
    detections = network.forward()

    face_roi = None
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_min:
            bbox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = bbox.astype("int")

            if (start_x < 0 or start_y < 0 or end_x > w or end_y > h):
                continue

            face_roi = orig_frame[start_y:end_y, start_x:end_x]
            face_roi = cv2.resize(face_roi, (90, 120))  ## comment IF you don`t need to resize all faces to a fixed size
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)  # draw bounding box
            if show_conf:
                text_conf = "{:.2f}%".format(confidence * 100)
                cv2.putText(frame, text_conf, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return face_roi, frame


# ==============================================================================
# TKINTER INTEGRATION
# ==============================================================================

class WebcamApp:
    def __init__(self, root, cam, detector_type, network_or_detector, max_width,
                 max_samples, starting_sample_number, person_name, final_path, final_path_full):
        self.root = root
        self.root.title("Captura de Face")
        self.root.geometry("800x650")  # Adjusted size to fit instruction label better
        self.root.resizable(False, False)

        # Store webcam and detector objects/settings
        self.cam = cam
        self.detector_type = detector_type
        self.network_or_detector = network_or_detector
        self.max_width = max_width
        self.max_samples = max_samples
        self.starting_sample_number = starting_sample_number
        self.person_name = person_name
        self.final_path = final_path
        self.final_path_full = final_path_full

        self.sample = 0
        self.current_face_roi = None
        self.current_processed_frame = None

        # Check if webcam opened successfully
        if not self.cam.isOpened():
            messagebox.showerror("Erro",
                                 "Não foi possível acessar a webcam. Verifique se está conectada e não está em uso.")
            self.root.destroy()
            return

        # Create a label to display the webcam feed
        self.video_label = tk.Label(self.root, bg="#D3D3D3")
        self.video_label.pack(pady=20, expand=True)

        # Create a label for instructions
        self.instruction_label = ttk.Label(self.root, text='Aperte "Q" para salvar', font=("Arial", 20))
        self.instruction_label.pack(pady=10)

        # Bind the 'q' key to the save_frame function
        self.root.bind('<q>', self.save_current_frame)
        self.root.bind('<Q>', self.save_current_frame)  # Also bind for uppercase Q

        # Set up a protocol for when the window is closed
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start updating the video feed
        self.update_feed()

    def update_feed(self):
        """Reads a frame from the webcam, processes it with face detection, and updates the Tkinter label."""
        ret, frame = self.cam.read()

        if ret:
            # Apply resizing if max_width is specified
            if self.max_width is not None:
                video_width, video_height = resize_video(frame.shape[1], frame.shape[0], self.max_width)
                frame = cv2.resize(frame, (video_width, video_height))

            # Perform face detection using your chosen detector
            if self.detector_type == "ssd":
                face_roi, processed_frame = detect_face_ssd(self.network_or_detector, frame)
            else:  # haarcascade
                face_roi, processed_frame = detect_face(self.network_or_detector, frame)

            self.current_face_roi = face_roi  # Store for saving
            self.current_processed_frame = processed_frame  # Store for saving

            # Convert the OpenCV BGR image to RGB for Tkinter display
            cv2_image_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            pil_image = Image.fromarray(cv2_image_rgb)

            # Resize the image to fit the label without distortion if the label's pack is expand=True
            # For simplicity and to fit fixed frame, we can just use the original frame size or set a fixed label size.
            # Here, we'll let Tkinter display the processed_frame as is if it fits.
            # If the frame is too large, you might want to scale it down.
            # For now, let's assume `max_width` handles the primary scaling.

            # To ensure the image fits the label, you might rescale it to the label's actual size
            # or predefine the label's size. Let's make it more robust:
            label_w = self.video_label.winfo_width()
            label_h = self.video_label.winfo_height()
            if label_w == 1 and label_h == 1:  # Initial state before widget fully rendered
                label_w, label_h = 640, 480  # Default size if label not yet rendered

            # Keep aspect ratio when fitting to label
            img_w, img_h = pil_image.size
            if img_w > label_w or img_h > label_h:
                ratio = min(label_w / img_w, label_h / img_h)
                new_w = int(img_w * ratio)
                new_h = int(img_h * ratio)
                pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)

            # Convert to Tkinter PhotoImage
            tk_image = ImageTk.PhotoImage(image=pil_image)

            # Update the label with the new image
            self.video_label.config(image=tk_image)
            self.video_label.image = tk_image  # Keep a reference to prevent garbage collection
        else:
            print("Failed to grab frame")

        # Schedule the next update after 10 milliseconds
        self.root.after(10, self.update_feed)

    def save_current_frame(self, event=None):
        """
        Saves the current detected face ROI and the full processed frame
        to respective folders.
        """
        if self.current_face_roi is not None and self.current_processed_frame is not None:
            self.sample += 1
            if self.sample > self.max_samples:
                messagebox.showinfo("Concluído",
                                    f"Número máximo de amostras ({self.max_samples}) atingido. Encerrando.")
                self.on_closing()
                return

            photo_sample = self.sample + self.starting_sample_number - 1 if self.starting_sample_number > 0 else self.sample
            image_name = self.person_name + "." + str(photo_sample) + ".jpg"

            # Save the cropped face (ROI)
            cv2.imwrite(os.path.join(self.final_path, image_name), self.current_face_roi)
            # Save the full image too (not cropped)
            cv2.imwrite(os.path.join(self.final_path_full, image_name), self.current_processed_frame)

            print(f"=> Foto {self.sample} salva como {image_name}")
            messagebox.showinfo("Sucesso", f"Foto {self.sample} salva: {image_name}")

        else:
            messagebox.showwarning("Aviso", "Nenhum rosto detectado ou frame disponível para salvar.")

    def on_closing(self):
        """Releases the webcam and closes the window."""
        if self.cam.isOpened():
            self.cam.release()  # Release the webcam
        self.root.destroy()  # Close the Tkinter window
        # Exit the script
        import sys
        sys.exit()  # Important to exit the whole script, not just the Tkinter window


# ==============================================================================
# MAIN APPLICATION SETUP 
# ==============================================================================
if __name__ == "__main__":
    ### Choose the face detector
    detector = "ssd"  # we suggest to keep SSD for more accurate detections
    max_width = 800  # leave None if you don't want to resize and want to keep the original size of the video stream frame

    max_samples = 5  # to control how many photos we'll be taking
    starting_sample_number = 0  # default=0

    if detector == "ssd":
        # For Face Detection with SSD (OpenCV's DNN) -> load weights from caffemodel
        try:
            network = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
        except cv2.error as e:
            print(f"Erro ao carregar o modelo SSD: {e}")
            print(
                "Certifique-se de que 'deploy.prototxt.txt' e 'res10_300x300_ssd_iter_140000.caffemodel' estão no mesmo diretório.")
            exit()
        network_or_detector = network
    else:
        # For Face Detection with HAAR CASCADE -> import haar cascade for face detection
        try:
            face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        except cv2.error as e:
            print(f"Erro ao carregar o Haar Cascade: {e}")
            print("Certifique-se de que 'haarcascade_frontalface_default.xml' está no mesmo diretório.")
            exit()
        network_or_detector = face_detector

    # video capture object
    cam = cv2.VideoCapture(0)

    folder_faces = "dataset/"  # where the cropped faces will be stored
    folder_full = "dataset_full/"  # where will be stored the full photos

    # The user need to type his name, so the faces will be saved in the proper subfolder
    person_name = input('Enter your name: ')
    person_name = parse_name(person_name)

    # Join the path (dataset directory + subfolder)
    final_path = os.path.sep.join([folder_faces, person_name])
    final_path_full = os.path.sep.join([folder_full, person_name])
    print("Todas as fotos serão salvas em {}".format(final_path))

    # Create the folders
    create_folders(final_path, final_path_full)

    root = tk.Tk()
    app = WebcamApp(root, cam, detector, network_or_detector, max_width,
                    max_samples, starting_sample_number, person_name, final_path, final_path_full)
    root.mainloop()

    # The following lines will only execute if root.mainloop() exits naturally,
    # but with sys.exit() in on_closing, they might not be reached.
    # We moved the cam.release() and cv2.destroyAllWindows() to on_closing.
    print("Completed!")