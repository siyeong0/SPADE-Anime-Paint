import cv2 as cv
import numpy as np

PALETTE = [
    (0,0,0),        # background
    (96,32,32),     # hair
    (255,255,255),  # eye
    (0,0,255),      # mouth
    (224,224,255),  # face
    (196,196,255),  # skin
    (196,0,196),    # cloth
]
PALETTE_RGB = [
    (0,0,0),        # background
    (32,32,96),     # hair
    (255,255,255),  # eye
    (255,0,0),      # mouth
    (255,224,224),  # face
    (255,196,196),  # skin
    (196,0,196),    # cloth
]
GRAY_PALETTE = [
    (0,0,0),
    (36,36,36),
    (73,73,73),
    (109,109,109),
    (146,146,146),
    (182,182,182),
    (219,219,219)
]

def img2seg(src):
    src = np.moveaxis(src, 0, 2)
    src = src.reshape(-1, 3)
    seg_list = []
    for color in PALETTE:
        seg_list.append(np.where(np.all(src==color, axis=1), 1.0, 0.0))
    dst = np.stack(seg_list,axis=1).reshape(256,256,7)
    
    return dst.astype(np.float32)

def seg2img(src):
    src = np.moveaxis(src,0,2)
    dst = [[PALETTE[np.argmax(val)] for val in buf]for buf in src]
    
    return np.array(dst).astype(np.uint8)

def color2gray(src):
    h,w,_ = src.shape
    src = src.reshape(-1, 3)
    dst = np.zeros(h*w).astype(np.float32)
    grays = [0.0,36.0,73.0,109.0,146.0,182.0,219.0]
    for color, gray in zip(PALETTE_RGB, grays):
        dst += np.where(np.all(src==color, axis=1), gray, 0.0)
    dst = dst.reshape((512,512))
    dst = np.stack((dst.copy(),dst.copy(),dst.copy()), axis=2)
    return dst.astype(np.uint8)
