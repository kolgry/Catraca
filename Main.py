import tkinter as tk
from tkinter import ttk

def on_entrar_click():
    print("Botão 'Entrar' clicado!")

def on_cadastro_click():
    print("Botão 'Cadastro' clicado!")

# Create the main window
root = tk.Tk()
root.title("Tela de Boas-Vindas")
root.geometry("600x400") # Set a reasonable size for the window
root.resizable(False, False) # Make the window not resizable for simplicity

# Create a frame to hold the buttons, this helps with centering
button_frame = ttk.Frame(root)
button_frame.pack(expand=True) # Center the frame in the window

# Style for the buttons
# You can customize these colors and fonts further
button_style = {
    "font": ("Arial", 24), # Larger font for the text
    "width": 10,          # Set a fixed width for the buttons
    "height": 3,          # Set a fixed height for the buttons
    "relief": "solid",    # Give them a solid border
    "borderwidth": 1      # Border width
}

# Create the "Entrar" button (light gray background)
entrar_button = tk.Button(button_frame, text="Entrar", command=on_entrar_click,
                          bg="#D3D3D3", # Light gray
                          fg="black",   # Black foreground (text)
                          **button_style)
entrar_button.pack(side=tk.LEFT, padx=50) # Pack to the left with some padding

# Create the "Cadastro" button (light purple background)
cadastro_button = tk.Button(button_frame, text="Cadastro", command=on_cadastro_click,
                            bg="#CCCCFF", # Light purple
                            fg="black",   # Black foreground (text)
                            **button_style)
cadastro_button.pack(side=tk.RIGHT, padx=50) # Pack to the right with some padding

# Run the Tkinter event loop
root.mainloop()