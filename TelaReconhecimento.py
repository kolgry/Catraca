# tela_reconhecimento.py (Atualizado)
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import pickle
import sys

from utils import resize_video

# Funções auxiliares (mantidas como estão)
def load_recognizer(option, training_data):
    if option == "eigenfaces":
        face_classifier = cv2.face.EigenFaceRecognizer_create()
    elif option == "fisherfaces":
        face_classifier = cv2.face.FisherFaceRecognizer_create()
    elif option == "lbph":
        face_classifier = cv2.face.LBPHFaceRecognizer_create()
    else:
        raise ValueError("The algorithms available are: Eigenfaces, Fisherfaces and LBPH")

    if not os.path.exists(training_data):
        raise FileNotFoundError(f"Arquivo de treinamento não encontrado: {training_data}")

    face_classifier.read(training_data)
    return face_classifier

def recognize_faces(network, face_classifier, orig_frame, face_names, threshold, conf_min=0.7):
    frame = orig_frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    network.setInput(blob)
    detections = network.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_min:
            bbox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = bbox.astype("int")

            if (start_x < 0 or start_y < 0 or end_x > w or end_y > h):
                continue

            face_roi = gray[start_y:end_y,start_x:end_x]
            face_roi = cv2.resize(face_roi, (90, 120))

            prediction, conf = face_classifier.predict(face_roi)

            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

            if prediction < len(face_names) and conf <= threshold:
                pred_name = face_names[prediction]
            else:
                pred_name = "Nao identificado"

            text = "{} -> {:.2f}".format(pred_name, conf)
            cv2.putText(frame, text, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

class TelaReconhecimento(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Parâmetros de configuração (mantidos como estão)
        self.recognizer_type = "lbph"
        self.training_data_path = "lbph_classifier.yml"
        self.face_names_path = "face_names.pickle"
        self.threshold = 10e5
        self.max_width = 800 # Usaremos isso para redimensionar o frame da webcam

        self.cam = None
        self.network = None
        self.face_classifier = None
        self.face_names = {}

        self.after_id = None

        # --- Novo Layout: Semelhante ao da TelaCapturaFace para consistência ---
        # Frame superior para o vídeo
        video_frame = tk.Frame(self, bg="#D3D3D3")
        # padx/pady são para a margem do frame em relação à janela principal.
        # expand=True permite que ele cresça.
        video_frame.pack(side="top", fill="both", expand=True, pady=10, padx=10)

        # O video_label vai preencher o video_frame
        self.video_label = tk.Label(video_frame, bg="#D3D3D3")
        self.video_label.pack(fill="both", expand=True) # expand=True aqui garante que o label preencha o frame

        # Frame inferior para os controles
        control_frame = tk.Frame(self, bg="#E0E0E0", bd=2, relief="groove")
        # side="bottom" para fixar no fundo, fill="x" para preencher a largura
        control_frame.pack(side="bottom", fill="x", pady=10, padx=10)

        self.status_label = ttk.Label(control_frame, text="Aguardando reconhecimento...", font=("Arial", 18))
        self.status_label.pack(pady=(10, 5)) # padding superior e inferior

        self.stop_button = tk.Button(control_frame, text="Parar Reconhecimento e Voltar",
                                     command=self.stop_recognition_and_return,
                                     font=("Arial", 16), bg="#FF6347", fg="white")
        self.stop_button.pack(pady=10) # padding superior e inferior

    def load_models(self):
        # ... (método load_models não precisa de alterações)
        try:
            self.network = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
        except cv2.error as e:
            messagebox.showerror("Erro de Modelo", f"Erro ao carregar o modelo SSD: {e}\nCertifique-se de que os arquivos 'deploy.prototxt.txt' e '.caffemodel' estão presentes.")
            self.controller.show_frame("TelaBoasVindas")
            return False

        try:
            self.face_classifier = load_recognizer(self.recognizer_type, self.training_data_path)
        except (ValueError, FileNotFoundError, cv2.error) as e:
            messagebox.showerror("Erro de Reconhecedor", f"Erro ao carregar o reconhecedor de faces: {e}\nCertifique-se de que '{self.training_data_path}' existe e é válido.")
            self.controller.show_frame("TelaBoasVindas")
            return False

        try:
            with open(self.face_names_path, "rb") as f:
                original_labels = pickle.load(f)
                self.face_names = {v: k for k, v in original_labels.items()}
        except FileNotFoundError:
            messagebox.showwarning("Aviso", f"Arquivo de nomes '{self.face_names_path}' não encontrado. O reconhecimento pode não funcionar corretamente.")
            self.face_names = {}
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar nomes das faces: {e}")
            self.controller.show_frame("TelaBoasVindas")
            return False
        return True

    def start_recognition(self):
        # ... (método start_recognition não precisa de alterações)
        if not self.load_models():
            return

        self.cam = cv2.VideoCapture(0)
        if not self.cam.isOpened():
            messagebox.showerror("Erro", "Não foi possível acessar a webcam. Verifique se está conectada e não está em uso.")
            self.controller.show_frame("TelaBoasVindas")
            return

        print("Iniciando reconhecimento facial...")
        self.status_label.config(text="Reconhecendo...")
        self.update_feed()

    def stop_recognition_and_return(self):
        # ... (método stop_recognition_and_return não precisa de alterações)
        if self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None
        if self.cam and self.cam.isOpened():
            self.cam.release()
            self.cam = None
        print("Reconhecimento parado.")
        self.controller.show_frame("TelaBoasVindas")

    def update_feed(self):
        """Lê o frame da webcam, processa e atualiza o display."""
        ret, frame = self.cam.read()

        if ret:
            # Redimensiona o frame da webcam para o max_width definido
            # O `resize_video` já calcula a altura proporcional.
            if self.max_width is not None:
                video_width, video_height = resize_video(frame.shape[1], frame.shape[0], self.max_width)
                frame = cv2.resize(frame, (video_width, video_height))

            # Processa o frame com reconhecimento
            processed_frame = recognize_faces(self.network, self.face_classifier, frame, self.face_names, self.threshold)

            cv2_image_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv2_image_rgb)

            # --- PARTE ALTERADA: REMOVER REDIMENSIONAMENTO BASEADO EM winfo_width/height ---
            # Deixe o PhotoImage lidar com o dimensionamento para preencher o Label.
            # O Label com pack(fill="both", expand=True) já se ajusta.
            tk_image = ImageTk.PhotoImage(image=pil_image)
            self.video_label.config(image=tk_image)
            self.video_label.image = tk_image # Mantenha a referência para evitar que o GC a remova

        self.after_id = self.after(10, self.update_feed)