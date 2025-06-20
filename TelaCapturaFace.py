# tela_captura_face.py (Atualizado)
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import os
import sys

# Importa as funções auxiliares do utils.py
# Certifique-se de que utils.py existe e contém essas funções ou as coloque aqui diretamente.
# Para este exemplo, vou assumir que utils.py existe e é importável.
from utils import resize_video, parse_name, create_folders, detect_face, detect_face_ssd


class TelaCapturaFace(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Parâmetros de configuração (mantidos como estão)
        self.detector_type = "ssd"
        self.max_width = 800
        self.max_samples = 10
        self.starting_sample_number = 0
        self.person_name = ""
        self.ra = ""  # Variável para armazenar o RA

        self.cam = None
        self.network_or_detector = None

        self.sample = 0
        self.current_face_roi = None
        self.current_processed_frame = None
        self.after_id = None

        # --- Novo Layout: Dividir em duas seções principais ---
        # Frame superior para o vídeo
        video_frame = tk.Frame(self, bg="#D3D3D3")
        video_frame.pack(side="top", fill="both", expand=True, pady=10, padx=10)  # Permite que o vídeo expanda

        self.video_label = tk.Label(video_frame, bg="#D3D3D3")
        self.video_label.pack(fill="both", expand=True)

        # Frame inferior para os controles e informações
        control_frame = tk.Frame(self, bg="#E0E0E0", bd=2, relief="groove")
        control_frame.pack(side="bottom", fill="x", pady=10, padx=10)  # Fill="x" para ocupar a largura total

        # Componentes dentro do control_frame
        self.instruction_label = ttk.Label(control_frame, text='Aperte "Q" para salvar', font=("Arial", 16))
        self.instruction_label.pack(pady=(10, 5))  # Adicionado padding superior

        self.sample_count_label = ttk.Label(control_frame, text="Amostras salvas: 0/10", font=("Arial", 14))
        self.sample_count_label.pack(pady=2)

        self.person_info_label = ttk.Label(control_frame, text="Nome: N/A | RA: N/A", font=("Arial", 14))
        self.person_info_label.pack(pady=(2, 10))  # Adicionado padding inferior

        # Botão para parar a captura e voltar
        self.stop_button = tk.Button(control_frame, text="Parar Captura e Voltar",
                                     command=self.stop_capture_and_return,
                                     font=("Arial", 16), bg="#FF6347", fg="white")
        self.stop_button.pack(pady=10)  # Garante espaço acima e abaixo do botão

    def load_detector(self):
        """Carrega o modelo do detector de face."""
        if self.detector_type == "ssd":
            try:
                self.network_or_detector = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt",
                                                                    "res10_300x300_ssd_iter_140000.caffemodel")
            except cv2.error as e:
                messagebox.showerror("Erro de Modelo",
                                     f"Erro ao carregar o modelo SSD: {e}\nCertifique-se de que 'deploy.prototxt.txt' e 'res10_300x300_ssd_iter_140000.caffemodel' estão no mesmo diretório.")
                self.controller.show_frame("TelaCadastro")
                return False
        else:  # haarcascade
            try:
                self.network_or_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            except cv2.error as e:
                messagebox.showerror("Erro de Modelo",
                                     f"Erro ao carregar o Haar Cascade: {e}\nCertifique-se de que 'haarcascade_frontalface_default.xml' está no mesmo diretório.")
                self.controller.show_frame("TelaCadastro")
                return False
        return True

    def start_capture(self, person_name, ra):
        """
        Inicia a captura da webcam com os dados da pessoa.
        Cria as pastas no formato "Nome_Sobrenome_RA".
        """
        self.person_name = parse_name(person_name)  # Já sanitiza o nome
        self.ra = ra
        self.sample = 0
        self.sample_count_label.config(text=f"Amostras salvas: {self.sample}/{self.max_samples}")
        self.person_info_label.config(text=f"Nome: {person_name} | RA: {ra}")

        if not self.load_detector():
            return

        self.cam = cv2.VideoCapture(0)
        if not self.cam.isOpened():
            messagebox.showerror("Erro",
                                 "Não foi possível acessar a webcam. Verifique se está conectada e não está em uso.")
            self.stop_capture_and_return()
            return

        # --- AQUI ESTÁ A ALTERAÇÃO CHAVE ---
        # Constrói o nome da pasta combinando nome e RA
        # Certifica-se de que 'self.ra' é uma string válida para evitar erros de concatenação
        folder_suffix = f"_{self.ra}" if self.ra else ""  # Adiciona _RA se o RA existir

        # O nome completo da pasta será "Nome_da_Pessoa_RA"
        full_folder_name = f"{self.person_name}{folder_suffix}"

        folder_faces = "dataset/"
        folder_full = "dataset_full/"

        # Usa o full_folder_name para criar os caminhos finais das pastas
        self.final_path = os.path.sep.join([folder_faces, full_folder_name])
        self.final_path_full = os.path.sep.join([folder_full, full_folder_name])

        create_folders(self.final_path, self.final_path_full)
        print(f"Todas as fotos para '{full_folder_name}' serão salvas em {self.final_path}")

        self.controller.bind('<q>', self.save_current_frame)
        self.controller.bind('<Q>', self.save_current_frame)

        self.update_feed()

    def stop_capture_and_return(self):
        """Para a captura e retorna à tela de cadastro."""
        if self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None
        if self.cam and self.cam.isOpened():
            self.cam.release()
            self.cam = None

        self.controller.unbind('<q>')
        self.controller.unbind('<Q>')

        self.controller.show_frame("TelaCadastro")

    def update_feed(self):
        ret, frame = self.cam.read()

        if ret:
            if self.max_width is not None:
                video_width, video_height = resize_video(frame.shape[1], frame.shape[0], self.max_width)
                frame = cv2.resize(frame, (video_width, video_height))

            if self.detector_type == "ssd":
                face_roi, processed_frame = detect_face_ssd(self.network_or_detector, frame)
            else:
                face_roi, processed_frame = detect_face(self.network_or_detector, frame)

            self.current_face_roi = face_roi
            self.current_processed_frame = processed_frame

            cv2_image_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv2_image_rgb)

            tk_image = ImageTk.PhotoImage(image=pil_image)
            self.video_label.config(image=tk_image)
            self.video_label.image = tk_image

        self.after_id = self.after(10, self.update_feed)

    def save_current_frame(self, event=None):
        if self.current_face_roi is not None and self.current_processed_frame is not None:
            if self.sample >= self.max_samples:
                messagebox.showinfo("Concluído", f"Número máximo de amostras ({self.max_samples}) atingido.")
                self.stop_capture_and_return()
                return

            self.sample += 1
            self.sample_count_label.config(text=f"Amostras salvas: {self.sample}/{self.max_samples}")

            photo_sample = self.sample + self.starting_sample_number - 1 if self.starting_sample_number > 0 else self.sample

            # Garante que o nome do arquivo também contenha o RA para consistência, embora não seja estritamente necessário para o treinamento
            # O nome da pasta já é suficiente para o treinamento extrair o RA.
            # No entanto, para fins de organização, incluir o RA no nome do arquivo pode ser útil.
            image_file_name = f"{self.person_name}_{self.ra}.{str(photo_sample)}.jpg" if self.ra else f"{self.person_name}.{str(photo_sample)}.jpg"

            cv2.imwrite(os.path.join(self.final_path, image_file_name), self.current_face_roi)
            cv2.imwrite(os.path.join(self.final_path_full, image_file_name), self.current_processed_frame)

            print(f"=> Foto {self.sample} salva como {image_file_name}")

            if self.sample == self.max_samples:
                messagebox.showinfo("Concluído",
                                    f"Número máximo de amostras ({self.max_samples}) atingido. Encerrando captura.")
                self.stop_capture_and_return()

        else:
            messagebox.showwarning("Aviso", "Nenhum rosto detectado ou frame disponível para salvar.")

