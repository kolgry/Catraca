import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import numpy as np
import cv2
from PIL import Image
import pickle
import re # Importar o módulo re para usar expressões regulares

class TelaBoasVindas(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        button_frame = ttk.Frame(self)
        button_frame.pack(expand=True, pady=20)

        button_style = {
            "font": ("Arial", 24),
            "width": 12,
            "height": 3,
            "relief": "solid",
            "borderwidth": 1
        }

        entrar_button = tk.Button(button_frame, text="Entrar",
                                  command=lambda: self.controller.show_recognition_screen(),
                                  bg="#D3D3D3",
                                  fg="black",
                                  **button_style)
        entrar_button.pack(side=tk.LEFT, padx=30)

        cadastro_button = tk.Button(button_frame, text="Cadastro",
                                    command=lambda: self.controller.show_frame("TelaCadastro"),
                                    bg="#CCCCFF",
                                    fg="black",
                                    **button_style)
        cadastro_button.pack(side=tk.RIGHT, padx=30)

        self.treinar_modelos_button = tk.Button(self, text="Treinar Modelos",
                                                 command=self.start_training_thread,
                                                 bg="#FFA07A",
                                                 fg="black",
                                                 **button_style)
        self.treinar_modelos_button.pack(pady=20)

        self.training_status_var = tk.StringVar()
        self.training_status_var.set("")
        status_label = ttk.Label(self, textvariable=self.training_status_var, font=("Arial", 14), foreground="blue")
        status_label.pack(pady=10)

    def start_training_thread(self):
        self.training_status_var.set("Iniciando treinamento... Por favor, aguarde.")
        self.treinar_modelos_button.config(state=tk.DISABLED)
        training_thread = threading.Thread(target=self._run_training)
        training_thread.daemon = True
        training_thread.start()

    def _run_training(self):
        training_path = 'dataset/'

        if not os.path.exists(training_path) or not os.listdir(training_path):
            self.after(0, lambda: self.training_status_var.set("Erro: O diretório 'dataset/' está vazio ou não existe."))
            self.after(0, lambda: messagebox.showerror("Erro de Treinamento",
                                                      "O diretório 'dataset/' não foi encontrado ou está vazio. "
                                                      "Certifique-se de que ele contém subpastas com as imagens de treinamento."))
            self.after(0, lambda: self.treinar_modelos_button.config(state=tk.NORMAL))
            return

        try:
            self.after(0, lambda: self.training_status_var.set("Carregando imagens do dataset..."))
            ids, faces, id_to_person_data = self._get_image_data(training_path) # Alterado para id_to_person_data

            if not faces:
                self.after(0, lambda: self.training_status_var.set("Erro: Nenhuma imagem de face válida encontrada para treinamento."))
                self.after(0, lambda: messagebox.showerror("Erro de Treinamento",
                                                          "Nenhuma imagem de face válida encontrada para treinamento. "
                                                          "Verifique o conteúdo do diretório 'dataset/'."))
                self.after(0, lambda: self.treinar_modelos_button.config(state=tk.NORMAL))
                return

            self.after(0, lambda: self.training_status_var.set("Salvando dados de faces no arquivo 'face_names.pickle'..."))
            # Salva o novo dicionário com ID -> {Nome, RA}
            with open("face_names.pickle", "wb") as f:
                pickle.dump(id_to_person_data, f)

            self.after(0, lambda: self.training_status_var.set("Treinando reconhecedor Eigenface..."))
            eigen_classifier = cv2.face.EigenFaceRecognizer_create()
            eigen_classifier.train(faces, ids)
            eigen_classifier.write('eigen_classifier.yml')

            self.after(0, lambda: self.training_status_var.set("Treinando reconhecedor Fisherface..."))
            fisher_classifier = cv2.face.FisherFaceRecognizer_create()
            fisher_classifier.train(faces, ids)
            fisher_classifier.write('fisher_classifier.yml')

            self.after(0, lambda: self.training_status_var.set("Treinando reconhecedor LBPH..."))
            lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
            lbph_classifier.train(faces, ids)
            lbph_classifier.write('lbph_classifier.yml')

            self.after(0, lambda: self.training_status_var.set("Treinamento concluído com sucesso!"))
            self.after(0, lambda: messagebox.showinfo("Treinamento", "Todos os modelos de reconhecimento facial foram treinados com sucesso!"))

        except Exception as e:
            self.after(0, lambda: self.training_status_var.set(f"Erro durante o treinamento: {e}"))
            self.after(0, lambda: messagebox.showerror("Erro de Treinamento", f"Ocorreu um erro durante o treinamento: {e}\nVerifique o console para mais detalhes."))
            print(f"Erro detalhado no treinamento: {e}")
        finally:
            cv2.destroyAllWindows()
            self.after(0, lambda: self.treinar_modelos_button.config(state=tk.NORMAL))

    def _get_image_data(self, path_train):
        """
        Função auxiliar para carregar dados de imagem do diretório de treinamento.
        Espera que os nomes das pastas estejam no formato "Nome_Sobrenome_RA".
        """
        # Adicionado o módulo 're' para usar expressões regulares na extração do nome e RA.
        # As pastas devem estar no formato: Nome_Sobrenome_RA (ex: 'Joao_Silva_RA12345')

        subdirs = [os.path.join(path_train, f) for f in os.listdir(path_train) if os.path.isdir(os.path.join(path_train, f))]
        faces = []
        ids = []
        id_to_person_data = {} # Dicionário para armazenar ID -> {"name": "Nome", "ra": "RA"}
        current_id_counter = 0 # Um contador para atribuir IDs sequenciais

        for subdir in subdirs:
            folder_name = os.path.split(subdir)[1] # Obtém o nome completo da pasta (ex: "Joao_Silva_RA12345")

            # Expressão regular para extrair o nome e o RA
            # Assume que o RA é uma sequência de caracteres alfanuméricos precedida por "RA" ou similar,
            # no final do nome da pasta, separada por underscore.
            match = re.match(r'(.+?)_([a-zA-Z0-9]+)$', folder_name)

            name = folder_name # Valor padrão, caso a regex não encontre o RA
            ra = "N/A" # Valor padrão para RA

            if match:
                # O grupo 1 é o nome (tudo antes do último underscore + RA)
                # O grupo 2 é o RA
                name_part = match.group(1).replace('_', ' ') # Substitui underscores no nome por espaços
                ra_part = match.group(2)

                name = name_part
                ra = ra_part
            else:
                # Se não houver correspondência com o padrão RA, usa o nome da pasta como nome e RA "N/A"
                print(f"Aviso: Não foi possível extrair o RA de '{folder_name}'. Usando 'N/A' para o RA.")


            # Atribui um ID para cada pessoa única
            person_found = False
            person_id = -1
            # Verifica se essa pessoa (nome+RA) já tem um ID atribuído
            for pid, pdata in id_to_person_data.items():
                if pdata["name"] == name and pdata["ra"] == ra:
                    person_id = pid
                    person_found = True
                    break

            if not person_found:
                person_id = current_id_counter
                id_to_person_data[person_id] = {"name": name, "ra": ra}
                current_id_counter += 1


            images_list = [os.path.join(subdir, f) for f in os.listdir(subdir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

            for path in images_list:
                try:
                    image = Image.open(path).convert('L')
                    face = np.array(image, 'uint8')
                    face = cv2.resize(face, (90, 120))

                    ids.append(person_id) # Adiciona o ID correspondente da pessoa
                    faces.append(face)

                    cv2.imshow("Training faces...", face)
                    cv2.waitKey(50)
                except Exception as e:
                    print(f"Erro ao carregar ou processar imagem {path}: {e}")
                    continue
        return np.array(ids), faces, id_to_person_data # Retorna o novo dicionário

