import json
import io
from commons import get_model, transform_image
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

model = get_model()
imagenet_class_index = json.load(open('imagenet_class_index.json'))

dataset_mean = [0.485, 0.456, 0.406]
dataset_std = [0.229, 0.224, 0.225]
mu = torch.Tensor((dataset_mean)).unsqueeze(-1).unsqueeze(-1)
std = torch.Tensor((dataset_std)).unsqueeze(-1).unsqueeze(-1)
unnormalize = lambda x: x*std + mu
normalize = lambda x: (x-mu)/std


def get_prediction(image_bytes):
    try:
        tensor = transform_image(image_bytes=image_bytes)
        outputs = model.forward(tensor)
    except Exception:
        return 0, 'error'
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


def get_perb(image_bytes):
    X = transform_image(image_bytes=image_bytes)

    U_perb = np.load('ImageNetGUAP_GoogLeNet_noise_0.9782.npy')
    U_perb = torch.from_numpy(U_perb).float()
    X_noise = unnormalize(X)+ U_perb
    perbimg  = torch.clamp(X_noise, 0, 1)
    tensor = normalize(perbimg)
    trans = transforms.ToPILImage()
    img = trans(perbimg.squeeze_(0)).convert("RGB")
    buffered = io.BytesIO()
    img.save(buffered, format="jpeg")
    buffered.seek(0)
    perb_image_bytes = buffered.getvalue()
    try:
        outputs = model.forward(tensor)
    except Exception:
        return 0, 0,'error'
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item()) 
    return imagenet_class_index[predicted_idx],perb_image_bytes

    