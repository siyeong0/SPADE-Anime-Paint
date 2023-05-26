from tkinter import *
from PIL import Image

from ui.canvas import canvas
from ui.imframe import imframe
from ui.menu import drawer_menu

from util.opt import inference_opt
from util.util import *
from util.converter import color2gray

class gen_image:
    def __init__(self):
        self.opt = inference_opt()
    
        self.generator = load_gen(self.opt, path='./weights/default_net_G.pth')
        self.encoder = load_enc(self.opt, path='./weights/default_net_E.pth')
        
        self.z = None
    
    def __call__(self, seg: Image, ref: Image, ref_dirty:bool = False):
        params = get_params(self.opt, seg.size)
        
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        seg = transform_label(seg) * 255.0
        transform_image = get_transform(self.opt, params)
        ref = transform_image(ref)
        
        data={
            'image' : ref.unsqueeze(dim=0),
            'label' : seg.unsqueeze(dim=0),
            'instance' : None
            }
        seg, ref = preprocess_input(self.opt, data)
        
        if ref_dirty:
            mu, var = self.encoder(ref)
            self.z = reparameterize(mu, var)
        
        result = self.generator(seg, self.z)
        
        image_numpy = result.squeeze(dim=0).detach().cpu().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        image_pil = Image.fromarray(image_numpy.astype(np.uint8))
        
        return image_pil

class app(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        self.draw_dirty = False
        self.ref_dirty = False
        self.generator = gen_image()
        # Initialize window
        self.title('Sementic Segment To Anime Illust')
        #self.geometry("1600x1000+100+100")
        self.frame = Frame(self, highlightbackground='black', highlightthickness=2)
        # Make image frame
        self.img_frame = Frame(self.frame)
        def dcf():
            self.draw_dirty = True
        self.mask_canvas = canvas(self.img_frame, text='Mask Canvas', load=True, save=True, callback=dcf)
        def lcf():
            self.ref_dirty = True
        self.ref_image = imframe(self.img_frame, image=Image.new('RGB', (512, 512)), text='Reference', load=True, load_callback=lcf)
        self.gen_image = imframe(self.img_frame, image=Image.new('RGB', (512, 512)), text='Generated', save=True)
        self.mask_canvas.pack(side='left')
        self.ref_image.pack(side='left')
        self.gen_image.pack(side='left')
        self.img_frame.pack(side='top')
        # Make menu frame
        self.edit_menu = drawer_menu(self.frame)
        self.edit_menu.pack(side='bottom')
        # Make reparameterize button
        def reparam():
            self.ref_dirty=True
        self.reparam_button = Button(self.frame, text='Rand',padx=10, pady=10, command=reparam)
        self.reparam_button.pack(side='right')
        # Packing
        self.frame.pack()
        # Update
        self.update_clock()

    def update_clock(self):
        # Update brush canvas's information
        self.mask_canvas.set_brush(self.edit_menu.get_brush_info())
        # Generate image
        if self.ref_dirty or self.draw_dirty:
            seg = Image.fromarray(color2gray(np.array(self.mask_canvas.image())))
            ref = self.ref_image.image()
            
            gen = self.generator(seg, ref, ref_dirty=self.ref_dirty)
            self.ref_dirty = False
            self.draw_dirty = False
            self.gen_image.update(image=gen)
        # call this function again in one second
        self.after(100, self.update_clock)
        
if __name__=='__main__':
    win = app()
    win.mainloop()

