#!/usr/bin/env python3
import cv2
import numpy as np
import subprocess
import tempfile
import os
import argparse

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Memory-efficient halftone video processor with multiple dithers")
parser.add_argument("--input", type=str, required=True, help="Input video file path")
parser.add_argument("--dither", type=str,
                    choices=["halftone","floyd-steinberg","atkinson","random",
                             "burkes","stucki","jjn","bayer8x8","clustered"],
                    default="halftone", help="Dither method")
parser.add_argument("--scale", type=int, default=2, help="Resolution reduction factor (2=1/4, 8=1/64)")
parser.add_argument("--color", type=str, choices=["red","green","blue","white"], default="green",
                    help="Color of the output dots")
args = parser.parse_args()

INPUT_VIDEO = args.input
DITHER_METHOD = args.dither
SCALE = args.scale
COLOR_NAME = args.color
OUTPUT_VIDEO = f"output_{DITHER_METHOD}_{COLOR_NAME}_{SCALE}x.mp4"
TEMP_VIDEO = tempfile.mktemp(suffix=".mp4")
CELL_SIZE = 4

# --- Color mapping ---
COLOR_MAP = {
    "red": (0,0,255),
    "green": (0,255,0),
    "blue": (255,0,0),
    "white": (255,255,255)
}
DOT_COLOR = COLOR_MAP[COLOR_NAME]

# --- Bayer matrices ---
BAYER_MATRIX_4x4 = np.array([
    [0, 8, 2,10],
    [12,4,14,6],
    [3,11,1,9],
    [15,7,13,5]
], dtype=np.uint8)
BAYER_THRESHOLD_4x4 = (BAYER_MATRIX_4x4 + 0.5) / 16.0 * 255

BAYER_MATRIX_8x8 = np.array([
    [0,32,8,40,2,34,10,42],
    [48,16,56,24,50,18,58,26],
    [12,44,4,36,14,46,6,38],
    [60,28,52,20,62,30,54,22],
    [3,35,11,43,1,33,9,41],
    [51,19,59,27,49,17,57,25],
    [15,47,7,39,13,45,5,37],
    [63,31,55,23,61,29,53,21]
], dtype=np.uint8)
BAYER_THRESHOLD_8x8 = (BAYER_MATRIX_8x8 + 0.5) / 64.0 * 255

# --- Dither functions ---
def halftone_dither(gray):
    h,w = gray.shape
    tiled = np.tile(BAYER_THRESHOLD_4x4,(h//CELL_SIZE+1,w//CELL_SIZE+1))[:h,:w]
    return (gray>tiled).astype(np.uint8)*255

def floyd_steinberg_dither(gray):
    img = gray.astype(np.float32)
    h,w = img.shape
    for y in range(h):
        for x in range(w):
            old = img[y,x]; new = 255 if old>127 else 0
            img[y,x]=new; err=old-new
            if x+1<w: img[y,x+1]+=err*7/16
            if y+1<h:
                if x>0: img[y+1,x-1]+=err*3/16
                img[y+1,x]+=err*5/16
                if x+1<w: img[y+1,x+1]+=err*1/16
    return np.clip(img,0,255).astype(np.uint8)

def atkinson_dither(gray):
    img = gray.astype(np.float32)
    h,w=img.shape
    for y in range(h):
        for x in range(w):
            old=img[y,x]; new=255 if old>127 else 0
            img[y,x]=new; err=(old-new)/8.0
            if x+1<w: img[y,x+1]+=err
            if x+2<w: img[y,x+2]+=err
            if y+1<h:
                if x>0: img[y+1,x-1]+=err
                img[y+1,x]+=err
                if x+1<w: img[y+1,x+1]+=err
            if y+2<h: img[y+2,x]+=err
    return np.clip(img,0,255).astype(np.uint8)

def random_dither(gray):
    noise=np.random.randint(0,256,gray.shape,dtype=np.uint8)
    return np.where(gray>noise,255,0).astype(np.uint8)

def burkes_dither(gray):
    # error diffusion similar to Floyd–Steinberg, simplified 3x5 kernel
    img = gray.astype(np.float32)
    h,w=img.shape
    for y in range(h):
        for x in range(w):
            old=img[y,x]; new=255 if old>127 else 0
            img[y,x]=new; err=old-new
            if x+1<w: img[y,x+1]+=err*8/32
            if x+2<w: img[y,x+2]+=err*4/32
            if y+1<h:
                if x-2>=0: img[y+1,x-2]+=err*2/32
                if x-1>=0: img[y+1,x-1]+=err*4/32
                img[y+1,x]+=err*8/32
                if x+1<w: img[y+1,x+1]+=err*4/32
                if x+2<w: img[y+1,x+2]+=err*2/32
    return np.clip(img,0,255).astype(np.uint8)

def stucki_dither(gray):
    img = gray.astype(np.float32)
    h,w=img.shape
    for y in range(h):
        for x in range(w):
            old=img[y,x]; new=255 if old>127 else 0
            img[y,x]=new; err=old-new
            # Stucki weights
            if x+1<w: img[y,x+1]+=err*8/42
            if x+2<w: img[y,x+2]+=err*4/42
            if y+1<h:
                for i in [-2,-1,0,1,2]:
                    if 0<=x+i<w: img[y+1,x+i]+=err*[2,4,8,4,2][i+2]/42
            if y+2<h:
                for i in [-2,-1,0,1,2]:
                    if 0<=x+i<w: img[y+2,x+i]+=err*[1,2,4,2,1][i+2]/42
    return np.clip(img,0,255).astype(np.uint8)

def jjn_dither(gray):
    img = gray.astype(np.float32)
    h,w=img.shape
    for y in range(h):
        for x in range(w):
            old=img[y,x]; new=255 if old>127 else 0
            img[y,x]=new; err=old-new
            # JJN 5x3 kernel
            kernel = np.array([[0,0,0,7,5],
                               [3,5,7,5,3],
                               [1,3,5,3,1]],dtype=float)/48
            for ky in range(3):
                for kx in range(-2,3):
                    nx=x+kx; ny=y+ky
                    if 0<=nx<w and 0<=ny<h: img[ny,nx]+=err*kernel[ky,kx+2]
    return np.clip(img,0,255).astype(np.uint8)

def bayer8x8_dither(gray):
    h,w=gray.shape
    tiled=np.tile(BAYER_THRESHOLD_8x8,(h//8+1,w//8+1))[:h,:w]
    return (gray>tiled).astype(np.uint8)*255

def clustered_dither(gray):
    # very simple clustered-dot, threshold 0-255
    h,w=gray.shape
    pattern=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=np.uint8)*255
    tiled=np.tile(pattern,(h//3+1,w//3+1))[:h,:w]
    return (gray>tiled).astype(np.uint8)*255

# --- Process frame ---
def process_frame(frame):
    small_frame = cv2.resize(frame,(frame.shape[1]//SCALE,frame.shape[0]//SCALE))
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    if DITHER_METHOD=="halftone": d = halftone_dither(gray)
    elif DITHER_METHOD=="floyd-steinberg": d = floyd_steinberg_dither(gray)
    elif DITHER_METHOD=="atkinson": d = atkinson_dither(gray)
    elif DITHER_METHOD=="random": d = random_dither(gray)
    elif DITHER_METHOD=="burkes": d = burkes_dither(gray)
    elif DITHER_METHOD=="stucki": d = stucki_dither(gray)
    elif DITHER_METHOD=="jjn": d = jjn_dither(gray)
    elif DITHER_METHOD=="bayer8x8": d = bayer8x8_dither(gray)
    elif DITHER_METHOD=="clustered": d = clustered_dither(gray)
    else: raise ValueError("Unknown dither")
    out_frame=np.zeros_like(small_frame)
    out_frame[d==255] = DOT_COLOR
    return out_frame

# --- Extract audio ---
print("[INFO] Extracting audio...")
subprocess.run(["ffmpeg","-y","-i",INPUT_VIDEO,"-vn","-acodec","copy","temp_audio.aac"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# --- Open video ---
cap=cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened(): raise IOError(f"Cannot open {INPUT_VIDEO}")
fps=cap.get(cv2.CAP_PROP_FPS)
orig_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width,height=orig_width//SCALE,orig_height//SCALE

fourcc=cv2.VideoWriter_fourcc(*"mp4v")
out=cv2.VideoWriter(TEMP_VIDEO,fourcc,fps,(width,height))

print(f"[INFO] Processing frames using {DITHER_METHOD} dithering in color {COLOR_NAME}...")

frame_count=0
while True:
    ret,frame=cap.read()
    if not ret: break
    processed=process_frame(frame)
    out.write(processed)
    frame_count+=1
    if frame_count%100==0: print(f"[INFO] Processed {frame_count} frames...")

cap.release()
out.release()

# --- Combine video + audio ---
print("[INFO] Muxing with audio...")
subprocess.run([
    "ffmpeg","-y","-i",TEMP_VIDEO,"-i","temp_audio.aac",
    "-c:v","copy","-c:a","aac","-map","0:v:0","-map","1:a:0",OUTPUT_VIDEO
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# --- Cleanup ---
os.remove("temp_audio.aac")
os.remove(TEMP_VIDEO)
print(f"[DONE] Saved {OUTPUT_VIDEO}")

