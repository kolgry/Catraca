import tkinter as tk
from tkinter import ttk

def on_cadastrar_click():
    """
    Function called when the 'Cadastrar' button is clicked.
    Retrieves the values from the input fields and prints them.
    """
    nome = nome_entry.get()
    ra = ra_entry.get()
    print(f"Nome: {nome}")
    print(f"RA: {ra}")
    # Add your logic here to process the 'nome' and 'ra' values
    # For example, save them to a database, perform validation, etc.

def validate_numeric_input(P):
    """
    Validation function to ensure that only numeric characters are entered.
    'P' is the value the entry field would have if the current edit is allowed.
    Returns True if the input is valid (numeric or empty), False otherwise.
    """
    if P.isdigit() or P == "":
        # If the input is all digits or an empty string, it's valid
        return True
    else:
        # Otherwise, it's invalid
        return False

# Create the main window
root = tk.Tk()
root.title("Tela de Cadastro")
root.geometry("600x400") # Set a reasonable size for the window
root.resizable(False, False) # Make the window not resizable for simplicity

# Create a frame to hold the input fields and button, helping with layout
form_frame = ttk.Frame(root)
form_frame.pack(expand=True, pady=50) # Center the frame and add vertical padding

# --- Nome Input Field ---
nome_label = ttk.Label(form_frame, text="Nome :", font=("Arial", 20))
nome_label.pack(anchor="w", padx=10, pady=(0, 5)) # Align left, add padding

nome_entry = ttk.Entry(form_frame, font=("Arial", 20), width=30)
nome_entry.pack(padx=10, pady=(0, 20)) # Add some space below the entry field

# --- RA Input Field ---
ra_label = ttk.Label(form_frame, text="RA :", font=("Arial", 20))
ra_label.pack(anchor="w", padx=10, pady=(0, 5)) # Align left, add padding

# Register the validation function with Tkinter
# This creates a Tcl command that can be passed to widgets
vcmd = root.register(validate_numeric_input)

ra_entry = ttk.Entry(form_frame,
                     font=("Arial", 20),
                     width=30,
                     validate="key",       # Validate on every key press
                     validatecommand=(vcmd, '%P')) # Pass the validation command and '%P'
ra_entry.pack(padx=10, pady=(0, 30)) # Add some space below the entry field

# --- Cadastrar Button ---
button_style = {
    "font": ("Arial", 24), # Larger font for the text
    "width": 12,           # Set a fixed width for the button
    "height": 2,           # Set a fixed height for the button
    "relief": "solid",     # Give it a solid border
    "borderwidth": 1       # Border width
}

cadastrar_button = tk.Button(form_frame, text="Cadastrar", command=on_cadastrar_click,
                             bg="#CCCCFF", # Light purple
                             fg="black",   # Black foreground (text)
                             **button_style)
cadastrar_button.pack(pady=10) # Add vertical padding

# Run the Tkinter event loop
root.mainloop()