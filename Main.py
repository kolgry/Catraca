# main_app.py (Atualizado com Janela Maior para Câmeras)
import tkinter as tk
from tkinter import ttk

# Importa as classes das suas telas
from BoasVindas import TelaBoasVindas
from TelaCadastro import TelaCadastro
from TelaCapturaFace import TelaCapturaFace
from TelaReconhecimento import TelaReconhecimento

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Aplicativo Integrado")
        self.geometry("600x550") # Tamanho inicial padrão
        self.resizable(False, False)

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (TelaBoasVindas, TelaCadastro, TelaCapturaFace, TelaReconhecimento):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.current_frame = None

        self.show_frame("TelaBoasVindas")

    def show_frame(self, page_name):
        frame = self.frames[page_name]

        if self.current_frame:
            if isinstance(self.current_frame, TelaCapturaFace) and self.current_frame.cam and self.current_frame.cam.isOpened():
                self.current_frame.stop_capture_and_return()
            elif isinstance(self.current_frame, TelaReconhecimento) and self.current_frame.cam and self.current_frame.cam.isOpened():
                self.current_frame.stop_recognition_and_return()

        frame.tkraise()
        self.current_frame = frame

        # Ajusta o tamanho da janela de acordo com a tela
        if page_name == "TelaCapturaFace" or page_name == "TelaReconhecimento":
            self.geometry("800x800") # <-- AUMENTADO A ALTURA AQUI PARA 800px
            self.resizable(False, False)
        else:
            self.geometry("600x550")
            self.resizable(False, False)

    def show_face_capture_screen(self, person_name, ra):
        capture_frame = self.frames["TelaCapturaFace"]
        self.show_frame("TelaCapturaFace")
        capture_frame.start_capture(person_name, ra)

    def show_recognition_screen(self):
        recognition_frame = self.frames["TelaReconhecimento"]
        self.show_frame("TelaReconhecimento")
        recognition_frame.start_recognition()

if __name__ == "__main__":
    app = App()
    app.mainloop()