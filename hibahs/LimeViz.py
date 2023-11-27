import numpy as np
import torch
from torchvision import transforms

from PIL import Image
from lime import lime_image

import torch.nn.functional as F

from skimage.segmentation import mark_boundaries

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'media/best_model_vitS16dino_unbal_checkpoint_epoch_6.pt'
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(device)

model = model.eval()

for param in model.parameters():
    param.requires_grad = False

model.blocks[-2:].requires_grad_(True)
model.norm.requires_grad_(True)
model = torch.nn.Sequential(model)
model = torch.nn.Sequential(model, torch.nn.AdaptiveAvgPool1d(768))
model = torch.nn.Sequential(model, torch.nn.Linear(768,4))
model = model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def initialize_model(cfg_data):
    if cfg_data["MODEL_ARCH"] == "resnet50":
        from models.ResNet50 import ResNet50_Model as net
        return net(cfg_data)
    if cfg_data["MODEL_ARCH"] == "dino_vits16":
        from models.DinoVits16 import DinoVits16_Model as dino
        return dino()
    return None
        

def get_image(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((224, 224)),
    ])

    return transf

def get_input_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    return transf

def get_input_tensors(img):
    transf = get_input_transform()
    return transf(img).unsqueeze(0).to(device) # unsqeeze converts single image to batch of 1


def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return transf

def batch_predict(images):
    model.eval()
    preprocess_transform = get_preprocess_transform()
    batch = torch.stack(tuple(preprocess_transform(img) for img in images), dim=0)
    model.to(device)
    batch = batch.to(device)
    logits = model(batch)
    probs = F.softmax(logits, dim=1)

    return probs.detach().cpu().numpy()

def batch_explaination(images):
    temps = []
    masks = []

    for img in images:
        explainer = lime_image.LimeImageExplainer()
        explaination = explainer.explain_instance(np.array(img),
                                                  batch_predict, # classification function
                                                  top_labels=4,
                                                  hide_color=0,
                                                  num_samples=1000) # number of images that will be sent to classification function
        temp, mask = explaination.get_image_and_mask(explaination.top_labels[0], 
                                                     positive_only=False,  
                                                     hide_rest=False)
        temps.append(temp)
        masks.append(mask)

    return temps, masks
