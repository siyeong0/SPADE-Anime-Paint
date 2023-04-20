from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import random
from functools import partial
from typing import Callable

class imframe(Frame):
    def __init__(self, master, 
                 image: Image, 
                 text: str = ' ', 
                 load: bool = False,
                 save: bool = False,
                 load_callback: Callable = None,
                 height=512, width=512, bd=5):
        super().__init__(master)
        self.height = height
        self.width = width
        self['height'] = height
        self['width'] = width
        self['bd'] = bd
        # Add Text
        self.text = Label(self, text = text)
        self.text.grid(row=0,column=0)
        # Add Image
        image =  image.resize((self.width, self.height))
        self.tk_image = ImageTk.PhotoImage(image)
        self.img_label = Label(self, image=self.tk_image)
        self.img_label.grid(row=1,column=0)
        # Add load/save button
        self.ls_frame = Frame(self)
        # Load 
        def load_fn(callback):
            file = filedialog.askopenfilename(initialdir='./',title = ' ',
                filetypes = [('All', '*.*'),('JPG files', '*.jpg'),('PNG files', '*.png'),])
            if file != '':
                self.update(Image.open(file))
            if callback != None:
                callback()
        self.load_button = Button(self.ls_frame,text='Load',command=partial(load_fn,load_callback)) if load else Label(self, text='')
        self.load_button.grid(row=0,column=0)
        # Add save button
        def save_fn():
            dir = filedialog.askdirectory()
            if dir != '':
                img = ImageTk.getimage(self.tk_image)
                name = random.randrange(10000,100000)
                img.save(dir+f'/{name}.png')
        self.save_button = Button(self.ls_frame,text='Save',command=save_fn) if save else Label(self, text='')
        self.save_button.grid(row=0,column=1)
        
        self.ls_frame.grid(row=2, column=0)
                
    def update(self, image: Image = None, text: str = None):
        if text != None:
            self.text['text'] = text
        if image != None:
            image = image.resize((self.width, self.height))
            self.tk_image = ImageTk.PhotoImage(image)
            self.img_label['image'] = self.tk_image
            
    def get_img_label(self):
        return self.img_label

    def image(self):
        return ImageTk.getimage(self.tk_image).convert('RGB')
