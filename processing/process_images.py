from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
from PIL import Image
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

# resize, and normalize
def scikit_preprocess(img_path):
    img = Image.open(img_path).convert("L")
    img = img.resize(size=(224,224))
    # img = np.array(img, dtype=np.float32) / 255.0
    return img

def torch_preprocess(img_path):
    print()

def resize():
    resize = Resize((224, 224))
