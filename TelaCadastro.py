# tela_cadastro.py (Atualizado e Verificado)
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# Importa a função de validação do utils.py
from utils import validate_numeric_input, parse_name

class TelaCadastro(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        form_frame = ttk.Frame(self)
        form_frame.pack(expand=True, pady=50)

        nome_label = ttk.Label(form_frame, text="Nome :", font=("Arial", 20))
        nome_label.pack(anchor="w", padx=10, pady=(0, 5))

        self.nome_entry = ttk.Entry(form_frame, font=("Arial", 20), width=30)
        self.nome_entry.pack(padx=10, pady=(0, 20))

        ra_label = ttk.Label(form_frame, text="RA :", font=("Arial", 20))
        ra_label.pack(anchor="w", padx=10, pady=(0, 5))

        vcmd = self.register(validate_numeric_input)

        self.ra_entry = ttk.Entry(form_frame,
                                  font=("Arial", 20),
                                  width=30,
                                  validate="key",
                                  validatecommand=(vcmd, '%P'))
        self.ra_entry.pack(padx=10, pady=(0, 30))

        button_style = {
            "font": ("Arial", 24),
            "width": 12,
            "height": 2,
            "relief": "solid",
            "borderwidth": 1
        }

        # Botão "Cadastrar"
        cadastrar_button = tk.Button(form_frame, text="Cadastrar", command=self.on_cadastrar_click,
                                     bg="#CCCCFF",
                                     fg="black",
                                     **button_style)
        cadastrar_button.pack(pady=10) # <-- Já tem 10px de padding vertical (acima e abaixo)

        # Botão "Voltar" - empacotado logo abaixo do "Cadastrar"
        back_button = tk.Button(form_frame, text="Voltar",
                                command=lambda: self.controller.show_frame("TelaBoasVindas"),
                                font=("Arial", 16),
                                bg="#ADD8E6",
                                fg="black")
        back_button.pack(pady=10) # <-- Este pady de 10px será a "margin-top" em relação ao "Cadastrar"
                                  #     e também dará espaço abaixo dele.


    def on_cadastrar_click(self):
        nome = self.nome_entry.get().strip()
        ra = self.ra_entry.get().strip()

        if not nome:
            messagebox.showwarning("Aviso", "Por favor, digite o nome.")
            return
        if not ra:
            messagebox.showwarning("Aviso", "Por favor, digite o RA.")
            return

        print(f"Nome: {nome}")
        print(f"RA: {ra}")

        self.controller.show_face_capture_screen(nome, ra)