# tela_boas_vindas.py (Atualizado)
import tkinter as tk
from tkinter import ttk

class TelaBoasVindas(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        button_frame = ttk.Frame(self)
        button_frame.pack(expand=True)

        button_style = {
            "font": ("Arial", 24),
            "width": 10,
            "height": 3,
            "relief": "solid",
            "borderwidth": 1
        }

        # Botão "Entrar" agora vai para a TelaReconhecimento
        entrar_button = tk.Button(button_frame, text="Entrar",
                                  command=lambda: self.controller.show_recognition_screen(), # <-- Alterado
                                  bg="#D3D3D3",
                                  fg="black",
                                  **button_style)
        entrar_button.pack(side=tk.LEFT, padx=50)

        cadastro_button = tk.Button(button_frame, text="Cadastro",
                                    command=lambda: self.controller.show_frame("TelaCadastro"),
                                    bg="#CCCCFF",
                                    fg="black",
                                    **button_style)
        cadastro_button.pack(side=tk.RIGHT, padx=50)

        # Você pode manter este botão, ou removê-lo se o "Entrar" for o principal.
        # detalhes_button = tk.Button(self, text="Ver Detalhes",
        #                             command=lambda: self.controller.show_frame("TelaDetalhes"),
        #                             font=("Arial", 18), bg="#A2D9CE", pady=5)
        # detalhes_button.pack(pady=20)