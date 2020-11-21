import json
import io
from commons import get_model, transform_image,flow_st,norm_ip
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms


imagenet_class_index = json.load(open('imagenet_class_index.json'))

dataset_mean = [0.485, 0.456, 0.406]
dataset_std = [0.229, 0.224, 0.225]
mu = torch.Tensor((dataset_mean)).unsqueeze(-1).unsqueeze(-1)
std = torch.Tensor((dataset_std)).unsqueeze(-1).unsqueeze(-1)
unnormalize = lambda x: x*std + mu
normalize = lambda x: (x-mu)/std


def get_prediction(image_bytes,modelname):
    model = get_model(modelname)
    try:
        tensor = transform_image(image_bytes=image_bytes)
        outputs = model.forward(tensor)
    except Exception:
        return 0, 'error'
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


def get_perb(image_bytes,modelname,basemodel):
    model = get_model(modelname)
    X = transform_image(image_bytes=image_bytes)

    if basemodel == 'VGG16':
        U_perb = np.load('perturbations/ImageNetGUAP_VGG16_noise_0.9847.npy')
        flow_field = np.load('perturbations/ImageNetGUAP_VGG16_flow_0.9847.npy')
    elif basemodel =='VGG19':
        U_perb = np.load('perturbations/ImageNetGUAP_VGG19_noise_0.9924.npy')
        flow_field = np.load('perturbations/ImageNetGUAP_VGG19_flow_0.9924.npy')
    elif basemodel == 'ResNet152':
        U_perb = np.load('perturbations/ImageNetGUAP_ResNet152_noise_0.9903.npy')
        flow_field = np.load('perturbations/ImageNetGUAP_ResNet152_flow_0.9903.npy')
    elif basemodel == 'GoogleNet':
        U_perb = np.load('perturbations/ImageNetGUAP_GoogLeNet_noise_0.9782.npy')
        flow_field = np.load('perturbations/ImageNetGUAP_GoogLeNet_flow_0.9782.npy')
    
    flow_field = torch.from_numpy(flow_field).float()
    U_perb = torch.from_numpy(U_perb).float()
    X_st = flow_st(unnormalize(X),flow_field,1)
    st_perb = X_st - unnormalize(X)

    st_perb = norm_ip(st_perb.squeeze(0)).unsqueeze(0)

    stimg  = torch.clamp(X_st, 0, 1)
    
    X_noise = X_st+ U_perb
    perbimg  = torch.clamp(X_noise, 0, 1)
    tensor = normalize(perbimg)
    trans = transforms.ToPILImage()

    img = trans(perbimg.squeeze_(0)).convert("RGB")
    buffered = io.BytesIO()
    img.save(buffered, format="png")
    buffered.seek(0)
    perb_image_bytes = buffered.getvalue()

    stimg = trans(stimg.squeeze_(0)).convert("RGB")
    buffered = io.BytesIO()
    stimg.save(buffered, format="png")
    buffered.seek(0)
    st_image_bytes = buffered.getvalue()

    st_perb = trans(st_perb.squeeze_(0)).convert("RGB")
    buffered = io.BytesIO()
    st_perb.save(buffered, format="png")
    buffered.seek(0)
    stperb_image_bytes = buffered.getvalue()

    noise = norm_ip(U_perb.squeeze(0)).unsqueeze(0) 
    noise = trans(noise.squeeze_(0)).convert("RGB")
    buffered = io.BytesIO()
    noise.save(buffered, format="png")
    buffered.seek(0)
    noise_image_bytes = buffered.getvalue()




    try:
        outputs = model.forward(tensor)
    except Exception:
        return 0, 0,'error'
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item()) 
    return imagenet_class_index[predicted_idx],st_image_bytes,perb_image_bytes,stperb_image_bytes,noise_image_bytes

    