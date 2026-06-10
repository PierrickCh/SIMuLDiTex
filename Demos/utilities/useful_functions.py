from PIL import Image
import os
import re
import numpy as np
import torchvision.transforms as transforms
import torch

def load_image_tensor(path:str, device:str='cuda', size:tuple=None, scaling_factor:int=None):
    img = Image.open(path).convert('RGB')
    numpy_img = np.array(img)
    h, w, c = numpy_img.shape
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    if scaling_factor is not None:
        img = img.resize((w*scaling_factor,h*scaling_factor), Image.BICUBIC)
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    return x

def load_array_tensor(array:np.array, device:str='cuda', size:tuple=None, scaling_factor:int=None):
    img = Image.fromarray(array).convert('RGB')
    numpy_img = np.array(array)
    h, w, c = numpy_img.shape
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    if scaling_factor is not None:
        img = img.resize((w*scaling_factor,h*scaling_factor), Image.BICUBIC)
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    return x

def clear_cuda_cache():
    torch.cuda.empty_cache()
    print("cleared cache")

# SIMuLDiTex specific functions

def n_params(model):
    pp=0
    for p in list(model.parameters(True)):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_latest_model_index(directory):
    files = os.listdir(directory)
    model_files = [f for f in files if re.match(r'model-(\d+)\.pt', f)]
    indices = []
    for model in model_files:
        match = re.match(r'model-(\d+)\.pt', model)
        if match:
            indices.append(int(match.group(1)))
    if indices:
        return max(indices)
    else:
        return None