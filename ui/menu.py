from tkinter import *
from functools import partial

def _from_rgb(rgb):
    return "#%02x%02x%02x" % rgb 

def _fg(rgb):
    s = sum(rgb)
    return 'black' if s > 200 else 'white'

class drawer_menu(Frame):
    def __init__(self, master, height=360, width=256):
        # Values
        self.brush_size_val = IntVar()
        self.brush_class = 'background'
        
        # Widgets
        super().__init__(master,padx=5,pady=3,height=height, width=width, highlightbackground='black', highlightthickness=2)
        # Current brush widget
        self.cb_frame = Frame(self)
        self.curr_brush_text = Label(self.cb_frame, text='Current Brush', height=1)
        self.curr_brush_tile = Label(self.cb_frame, bg='black', height=1, width=2)
        self.curr_brush_text.grid(row=0,column=0)
        self.curr_brush_tile.grid(row=0,column=1)
        # Brush thicknes widget
        self.br_scale_frame = Frame(self)
        self.brush_size_text = Label(self.br_scale_frame, text='Brush Thickness')
        self.brush_size_widget = Scale(self.br_scale_frame, variable=self.brush_size_val, from_=5, to=55, orient=HORIZONTAL)
        self.brush_size_text.grid(row=0,column=0)
        self.brush_size_widget.grid(row=0,column=1)
        # Brush color widget
        self.brush_colors_frame = Frame(self)
        dict = {
            'background':   (0,0,0),
            'hair':         (32,32,96),
            'eye':          (255,255,255),
            'mouth':        (255,0,0),
            'face':         (255,224,224),
            'skin':         (255,196,196),
            'cloth':        (196,0,196),
        }
        def set_bc(bc:str):
            self.brush_class = bc
            self.curr_brush_tile['bg']=_from_rgb(dict[bc])
        self.brushes = [Button(self.brush_colors_frame, text=key, bg=_from_rgb(val), fg=_fg(val), height=1,width=10, command = partial(set_bc, key)) for key, val in dict.items()]
        for idx, br in enumerate(self.brushes):
            br.grid(row=0,column=idx)
        
        # Place widgets
        self.cb_frame.grid(row=0, column=0, padx = 10)
        self.br_scale_frame.grid(row=1, column=0, padx = 10)
        self.brush_colors_frame.grid(row=1, column=1, padx = 10)
        
    def get_brush_info(self):
        return (self.brush_class, self.brush_size_val.get())
