from tkinter import *
from tkinter import filedialog

class Misc:
    def __init__(self):
        self.version = "0.0.1"
        self.author = "Mini Project Group 12"
        self.description = "Miscellaneous functions"
    
    def get_version(self):
        return self.version

    def open_file(self):
        file_path = filedialog.askopenfilename()
        return file_path
