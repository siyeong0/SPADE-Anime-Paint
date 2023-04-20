import cv2 as cv
import numpy as np
from PIL import Image
from PIL import ImageTk
from typing import Callable

from ui.imframe import imframe

class canvas(imframe):
    def __init__(self, master, 
                 text: str = ' ', 
                 load: bool = False,
                 save: bool = False,
                 callback: Callable = None,
                 height=512, width=512, bd=5):
        self.canvas = None
        self.clear()   
        super().__init__(master, image=self.get('im'), text=text, load=load, save=save, height=height, width=width, bd=bd, load_callback=callback)
          
        self.palette = None
        self.reset_palette()
            
        #self.brush_class = 'background'
        self.brush_class = 'hair'
        self.brush_radius = 5
        
        self.prev_xy = (-1,-1)
        self.is_left_clicked = False
        def left_click(event):
            self.is_left_clicked = True
            self.prev_xy = (event.x, event.y)
        def left_release(event):
            self.is_left_clicked = False
            self.prev_xy = (-1,-1)
        def draw(event):
            if callback!= None:
                callback()
            if self.is_left_clicked:
                cv.line(self.canvas, self.prev_xy, (event.x, event.y), self.palette[self.brush_class], self.brush_radius)
                self.update(image=self.get('im'))
                self.prev_xy = (event.x, event.y)
        super().get_img_label().bind('<Button-1>', left_click)
        super().get_img_label().bind('<ButtonRelease-1>', left_release)
        super().get_img_label().bind('<B1-Motion>', draw)
        
        
    def update(self, image: Image = None, text: str = None):
        super().update(image, text)
        self.canvas = np.array(self.image())[:, :, ::-1].copy()
        
    def clear(self):
        self.canvas = np.zeros((512,512,3)).astype(np.uint8)
        
    def get(self, mode='im'):
        buf = cv.cvtColor(self.canvas, cv.COLOR_BGR2RGB).copy()
        if mode=='im':
            img = Image.fromarray(buf)
            return img
        elif mode=='cv':
            return buf
        elif mode=='tk':
            buf = Image.fromarray(buf)
            return ImageTk.PhotoImage(buf)
        
    def reset_palette(self):
        # BGR
        self.palette = {
            'background':   (0,0,0),
            'hair':         (96,32,32),
            'eye':          (255,255,255),
            'mouth':        (0,0,255),
            'face':         (224,224,255),
            'skin':         (196,196,255),
            'cloth':        (196,0,196),
        }
    
    def set_brush(self, brush_info):
        self.brush_class = brush_info[0]
        self.brush_radius = brush_info[1]
    
        