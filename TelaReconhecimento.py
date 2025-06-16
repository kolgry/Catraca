import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import pickle
import sys
import time  # Importado para usar funções de tempo


# --- Funções Auxiliares ---

def resize_video(original_width, original_height, max_width):
    """
    Redimensiona as dimensões de um vídeo para que a largura máxima não seja excedida,
    mantendo a proporção.
    """
    if original_width > max_width:
        ratio = max_width / original_width
        new_width = max_width
        new_height = int(original_height * ratio)
    else:
        new_width = original_width
        new_height = original_height
    return new_width, new_height


def load_recognizer(option, training_data):
    """
    Carrega o reconhecedor facial com base na opção e nos dados de treinamento.
    """
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


def recognize_faces(network, face_classifier, orig_frame, id_to_person_data, threshold, conf_min=0.7):
    """
    Realiza o reconhecimento facial em um frame.
    Desenha retângulos e nomes sobre as faces detectadas.
    Retorna o frame processado e os dados da pessoa identificada (nome e RA),
    ou um dicionário indicando "Nao identificado" se ninguém for reconhecido.
    """
    frame = orig_frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    network.setInput(blob)
    detections = network.forward()

    # Inicializa com dados de "Nao identificado"
    identified_person_data_in_frame = {"name": "Nao identificado", "ra": ""}

    # Flag para verificar se alguma face válida foi detectada e identificada
    face_identified_in_current_frame = False

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_min:
            bbox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = bbox.astype("int")

            # Garante que as coordenadas da face estejam dentro dos limites do frame
            if (start_x < 0 or start_y < 0 or end_x > w or end_y > h):
                continue

            face_roi = gray[start_y:end_y, start_x:end_x]
            face_roi = cv2.resize(face_roi, (90, 120))  # Redimensiona para o tamanho esperado pelo classificador

            prediction, conf = face_classifier.predict(face_roi)

            # Desenha o retângulo no frame
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

            pred_name_display = "Nao identificado"
            pred_ra_display = ""

            # Verifica se a previsão está dentro dos IDs conhecidos e a confiança é suficiente
            if prediction in id_to_person_data and conf <= threshold:
                person_info = id_to_person_data[prediction]
                pred_name_display = person_info.get("name", "Desconhecido")  # Obtém o nome ou "Desconhecido"
                pred_ra_display = person_info.get("ra", "")  # Obtém o RA ou vazio

                # Atualiza os dados da pessoa identificada no frame (prioriza a última identificação)
                identified_person_data_in_frame = {"name": pred_name_display, "ra": pred_ra_display}
                face_identified_in_current_frame = True  # Uma face foi identificada com sucesso

            # Adiciona o texto no frame (Nome -> Confiança, e RA se disponível)
            text = "{} -> {:.2f}".format(pred_name_display, conf)
            if pred_ra_display:
                text += f" (RA: {pred_ra_display})"  # Adiciona o RA ao texto exibido

            cv2.putText(frame, text, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Se nenhuma face foi identificada com sucesso neste frame, retorne "Nao identificado"
    if not face_identified_in_current_frame:
        return frame, {"name": "Nao identificado", "ra": ""}

    return frame, identified_person_data_in_frame


# --- Classe da Tela de Reconhecimento Facial (com pop-up) ---
class TelaReconhecimento(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Parâmetros de configuração para o reconhecimento
        self.recognizer_type = "lbph"
        self.training_data_path = "lbph_classifier.yml"
        self.face_names_path = "face_names.pickle"
        self.threshold = 10e5  # Limite de confiança para identificação
        self.max_width = 800  # Largura máxima do feed de vídeo

        # Variáveis de estado para a webcam e modelos
        self.cam = None
        self.network = None
        self.face_classifier = None
        self.id_to_person_data = {}  # Dicionário para mapear ID -> {"name": "Nome", "ra": "RA"}

        # Variáveis de controle do loop do Tkinter e pop-up
        self.after_id = None
        self.popup_window = None  # Referência para a janela Toplevel do pop-up
        self.is_popup_active = False  # Flag para controlar se o pop-up está visível

        # Variáveis para a lógica de delay de 1.5 segundos
        self.last_identified_data_consistent = None  # Armazena os dados da pessoa identificada consistentemente
        self.identification_start_time = None  # time.time() quando a identificação consistente começou
        self.recognition_delay_threshold = 1.5  # Delay em segundos antes de mostrar o pop-up

        # --- Estrutura da UI ---
        # Frame superior para o feed de vídeo
        video_frame = tk.Frame(self, bg="#D3D3D3")
        video_frame.pack(side="top", fill="both", expand=True, pady=10, padx=10)

        self.video_label = tk.Label(video_frame, bg="#D3D3D3")
        self.video_label.pack(fill="both", expand=True)

        # Frame inferior para os controles (status e botão)
        control_frame = tk.Frame(self, bg="#E0E0E0", bd=2, relief="groove")
        control_frame.pack(side="bottom", fill="x", pady=10, padx=10)

        # Rótulo de status
        self.status_label = ttk.Label(control_frame, text="Aguardando reconhecimento...", font=("Arial", 18))
        self.status_label.pack(pady=(10, 5))

        # Botão para parar reconhecimento e voltar
        self.stop_button = tk.Button(control_frame, text="Parar Reconhecimento e Voltar",
                                     command=self.stop_recognition_and_return,
                                     font=("Arial", 16), bg="#FF6347", fg="white")
        self.stop_button.pack(pady=10)

    def load_models(self):
        """Carrega os modelos de detecção e reconhecimento facial."""
        try:
            # Caminho para os modelos DNN do OpenCV (assegure-se de que estão na pasta correta)
            prototxt_path = "deploy.prototxt.txt"
            caffemodel_path = "res10_300x300_ssd_iter_140000.caffemodel"

            if not os.path.exists(prototxt_path) or not os.path.exists(caffemodel_path):
                messagebox.showerror("Erro de Modelo",
                                     f"Arquivos de modelo DNN não encontrados:\n{prototxt_path}\n{caffemodel_path}\n"
                                     "Certifique-se de que estão na mesma pasta do script.")
                self._handle_error_return()  # Tenta retornar à tela anterior ou fechar
                return False

            self.network = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        except cv2.error as e:
            messagebox.showerror("Erro de Modelo",
                                 f"Erro ao carregar o modelo SSD: {e}\nCertifique-se de que os arquivos 'deploy.prototxt.txt' e '.caffemodel' estão presentes e são válidos.")
            self._handle_error_return()
            return False

        try:
            self.face_classifier = load_recognizer(self.recognizer_type, self.training_data_path)
        except (ValueError, FileNotFoundError, cv2.error) as e:
            messagebox.showerror("Erro de Reconhecedor",
                                 f"Erro ao carregar o reconhecedor de faces: {e}\nCertifique-se de que '{self.training_data_path}' existe e é válido (rode o treinamento).")
            self._handle_error_return()
            return False

        try:
            with open(self.face_names_path, "rb") as f:
                # Tenta carregar o formato de face_names.pickle
                loaded_data = pickle.load(f)

                # Verifica se o formato carregado é o antigo ({nome: id})
                if isinstance(loaded_data, dict) and all(
                        isinstance(k, str) and isinstance(v, int) for k, v in loaded_data.items()):
                    # Formato antigo: {nome: id}. Precisamos inverter para id: nome e adicionar RA
                    print("AVISO: Formato antigo de 'face_names.pickle' detectado. Gerando RAs fictícios.")
                    self.id_to_person_data = {
                        v: {"name": k, "ra": f"RA{v:05d}"}  # Gera um RA fictício (ex: RA00001)
                        for k, v in loaded_data.items()
                    }
                    messagebox.showwarning("Aviso de RA",
                                           "O arquivo 'face_names.pickle' não contém informações de RA. "
                                           "RAs fictícios foram gerados. Por favor, **atualize seu script de treinamento** "
                                           "para incluir RAs reais ao salvar 'face_names.pickle' no formato correto (ID -> {Nome, RA}).")
                # Verifica se o formato carregado é o novo desejado ({id: {name: nome, ra: ra}})
                elif isinstance(loaded_data, dict) and all(
                        isinstance(k, int) and isinstance(v, dict) and 'name' in v and 'ra' in v for k, v in
                        loaded_data.items()):
                    self.id_to_person_data = loaded_data
                # Verifica se o formato é {nome: {id: id, ra: ra}} e converte para id: {name: nome, ra: ra}
                elif isinstance(loaded_data, dict) and all(
                        isinstance(v, dict) and 'id' in v and 'ra' in v for v in loaded_data.values()):
                    self.id_to_person_data = {
                        data["id"]: {"name": name, "ra": data["ra"]}
                        for name, data in loaded_data.items()
                    }
                else:
                    raise ValueError("Formato desconhecido no arquivo 'face_names.pickle'.")

                if not self.id_to_person_data:
                    messagebox.showwarning("Aviso",
                                           f"Arquivo de nomes '{self.face_names_path}' carregado, mas está vazio ou inválido.")
                    self.id_to_person_data = {}  # Garante que seja um dicionário vazio

        except FileNotFoundError:
            messagebox.showwarning("Aviso",
                                   f"Arquivo de nomes '{self.face_names_path}' não encontrado. O reconhecimento pode não funcionar corretamente.")
            self.id_to_person_data = {}
        except Exception as e:
            messagebox.showerror("Erro",
                                 f"Erro ao carregar nomes das faces: {e}\nVerifique o formato do arquivo 'face_names.pickle'.")
            self._handle_error_return()
            return False
        return True

    def start_recognition(self):
        """Inicia a captura da webcam e o loop de reconhecimento."""
        if not self.load_models():  # Tenta carregar os modelos; se falhar, retorna
            return

        self.cam = cv2.VideoCapture(0)  # 0 para a câmera padrão
        if not self.cam.isOpened():
            messagebox.showerror("Erro",
                                 "Não foi possível acessar a webcam. Verifique se está conectada e não está em uso.")
            self._handle_error_return()
            return

        print("Iniciando reconhecimento facial...")
        self.status_label.config(text="Reconhecendo...")
        self.update_feed()  # Inicia o loop de atualização do feed

    def stop_recognition_and_return(self):
        """Para a captura da webcam e retorna à tela de boas-vindas ou fecha a janela."""
        if self.after_id:
            self.after_cancel(self.after_id)  # Cancela o próximo 'after' agendado
            self.after_id = None
        if self.cam and self.cam.isOpened():
            self.cam.release()  # Libera a câmera
            self.cam = None

        # Fecha qualquer pop-up de identificação ativo
        if self.popup_window:
            self.close_popup()

        # Reseta as variáveis de delay/consistência para o próximo início
        self.last_identified_data_consistent = None
        self.identification_start_time = None
        self.is_popup_active = False

        print("Reconhecimento parado.")
        self._handle_error_return()  # Chama a função auxiliar para retornar ou fechar

    def update_feed(self):
        """Lê o frame da webcam, processa e atualiza o display."""
        ret, frame = self.cam.read()
        current_time = time.time()  # Obtém o tempo atual em segundos

        if ret:
            # Redimensiona o frame para exibição
            if self.max_width is not None:
                video_width, video_height = resize_video(frame.shape[1], frame.shape[0], self.max_width)
                frame = cv2.resize(frame, (video_width, video_height))

            # Processa o frame com detecção e reconhecimento
            processed_frame, identified_person_data_in_frame = recognize_faces(
                self.network, self.face_classifier, frame, self.id_to_person_data, self.threshold
            )

            # --- Lógica de Delay de 1.5 segundos para o Pop-up ---
            if identified_person_data_in_frame["name"] != "Nao identificado":
                # Uma pessoa foi identificada neste frame
                current_person_data = identified_person_data_in_frame

                # Verifica se é uma NOVA identificação ou se a pessoa mudou
                if self.last_identified_data_consistent is None or \
                        self.last_identified_data_consistent["name"] != current_person_data["name"]:

                    # Reinicia o rastreamento para a nova pessoa
                    self.last_identified_data_consistent = current_person_data
                    self.identification_start_time = current_time
                    self.is_popup_active = False  # Garante que o pop-up será mostrado se a consistência for mantida

                    # Se um pop-up estava ativo para outra pessoa, fecha-o
                    if self.popup_window and self.popup_window.winfo_exists():
                        self.close_popup()

                else:
                    # A mesma pessoa foi identificada consistentemente
                    if not self.is_popup_active and \
                            (current_time - self.identification_start_time >= self.recognition_delay_threshold):
                        # Se o pop-up não está ativo E o tempo de delay passou, mostre o pop-up
                        self.show_identification_popup(self.last_identified_data_consistent)
                        self.is_popup_active = True
            else:
                # Nenhuma pessoa identificada (ou "Nao identificado")
                # Reseta o estado de identificação consistente
                self.last_identified_data_consistent = None
                self.identification_start_time = None

                # Se o pop-up estava ativo e a pessoa saiu do frame, feche-o
                if self.is_popup_active and self.popup_window and self.popup_window.winfo_exists():
                    self.close_popup()
                    self.is_popup_active = False  # Reseta a flag

            # Converte e exibe o frame processado no Tkinter Label
            cv2_image_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv2_image_rgb)
            tk_image = ImageTk.PhotoImage(image=pil_image)

            self.video_label.config(image=tk_image)
            self.video_label.image = tk_image  # Mantenha a referência para evitar que o GC a remova

        # Agenda a próxima atualização do feed
        self.after_id = self.after(10, self.update_feed)

    def show_identification_popup(self, person_data):
        """
        Cria e exibe uma janela pop-up informando que uma pessoa foi identificada.
        Recebe um dicionário com 'name' e 'ra' da pessoa.
        """
        if self.popup_window and self.popup_window.winfo_exists():  # Evita múltiplos pop-ups se já existir
            self.close_popup()

        self.popup_window = tk.Toplevel(self)
        self.popup_window.title("Pessoa Identificada!")
        self.popup_window.geometry("450x220")  # Tamanho ajustado para acomodar o RA

        # Centraliza o pop-up em relação à janela pai (ou à tela se estiver isolado)
        # Verifica se self.master existe e é uma janela ativa antes de usar winfo_x/y/width/height
        if self.master and self.master.winfo_exists():
            main_app_x = self.master.winfo_x()
            main_app_y = self.master.winfo_y()
            main_app_width = self.master.winfo_width()
            main_app_height = self.master.winfo_height()
        else:  # Fallback para quando a TelaReconhecimento é testada isoladamente
            main_app_x = self.winfo_x()
            main_app_y = self.winfo_y()
            main_app_width = self.winfo_width()
            main_app_height = self.winfo_height()

        popup_width = 450
        popup_height = 220
        center_x = main_app_x + main_app_width // 2 - popup_width // 2
        center_y = main_app_y + main_app_height // 2 - popup_height // 2
        self.popup_window.geometry(f"{popup_width}x{popup_height}+{center_x}+{center_y}")

        # Torna o pop-up modal e sempre no topo
        if self.master and self.master.winfo_exists():
            self.popup_window.transient(self.master)  # Garante que o pop-up fique acima da janela principal
        self.popup_window.grab_set()  # Captura todos os eventos para esta janela (torna-a modal)
        self.popup_window.wm_attributes("-topmost", True)  # Mantém a janela sempre no topo

        # Estilo visual para o pop-up
        self.popup_window.configure(bg="#E6F7FF")  # Azul claro suave
        self.popup_window.resizable(False, False)  # Impede redimensionamento manual

        # Mensagem de boas-vindas com o nome
        name_label = ttk.Label(self.popup_window, text=f"Bem-vindo(a), {person_data.get('name', 'Desconhecido')}!",
                               font=("Arial", 22, "bold"), foreground="#2C3E50", background="#E6F7FF")
        name_label.pack(pady=(20, 5), padx=20)  # Padding superior e inferior ajustados

        # Exibição do RA
        ra_label = ttk.Label(self.popup_window, text=f"RA: {person_data.get('ra', 'N/A')}",
                             font=("Arial", 18), foreground="#34495E", background="#E6F7FF")
        ra_label.pack(pady=(0, 15), padx=20)  # Padding inferior ajustado

        # Botão OK para fechar o pop-up
        close_button = tk.Button(self.popup_window, text="OK", command=self.close_popup,
                                 font=("Arial", 14), bg="#28A745", fg="white",  # Verde mais vibrante
                                 relief="raised", bd=2, width=8, height=1)
        close_button.pack(pady=10)
        close_button.bind("<Enter>", lambda e: close_button.config(bg="#218838"))  # Efeito de hover
        close_button.bind("<Leave>", lambda e: close_button.config(bg="#28A745"))

        # Configura o protocolo de fechamento da janela (botão 'X' da barra de título)
        self.popup_window.protocol("WM_DELETE_WINDOW", self.close_popup)

    def close_popup(self):
        """Fecha a janela pop-up e redefine as variáveis de estado."""
        if self.popup_window and self.popup_window.winfo_exists():
            self.popup_window.grab_release()  # Libera o "grab" de eventos para que a janela principal seja clicável
            self.popup_window.destroy()  # Destrói a janela
            self.popup_window = None
            self.is_popup_active = False  # Reseta a flag de pop-up ativo

    def _handle_error_return(self):
        """
        Função auxiliar para retornar à tela principal de boas-vindas
        ou fechar a janela se a TelaReconhecimento estiver sendo executada isoladamente.
        """
        if self.controller and hasattr(self.controller, 'show_frame'):
            self.controller.show_frame("TelaBoasVindas")
        else:  # Caso não haja um controller (ex: executando este arquivo diretamente), fecha a janela principal
            self.master.destroy()


# --- Exemplo de Execução Principal (para testar esta TelaReconhecimento isoladamente) ---
# Este bloco só será executado se você rodar este arquivo Python diretamente.
# Em uma aplicação maior (com 'App' principal), esta parte não é usada.
if __name__ == "__main__":
    class MockController:
        """Um controlador mock simples para simular a navegação para testes."""

        def show_frame(self, page_name):
            print(f"Simulando navegação para: {page_name}")
            # Se a 'navegação' for para a TelaBoasVindas, feche a janela principal para terminar o teste
            if page_name == "TelaBoasVindas":
                root.destroy()


    root = tk.Tk()
    root.title("Teste da Tela de Reconhecimento Facial")
    root.geometry("800x600")

    mock_controller = MockController()
    recognition_screen = TelaReconhecimento(parent=root, controller=mock_controller)
    recognition_screen.pack(fill="both", expand=True)

    # Inicia o reconhecimento automaticamente quando a tela de teste é carregada
    recognition_screen.start_recognition()

    root.mainloop()
