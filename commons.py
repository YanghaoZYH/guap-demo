import io


from PIL import Image
from torchvision import models
import torchvision.transforms as transforms
import torch
import torch.utils.model_zoo as model_zoo

model_urls = {
    'densenet121':'./checkpoints/densenet121-a639ec97.pth'
}


def get_model(modelname):
    if modelname == 'VGG16':
        model = models.vgg16(pretrained=True)
    elif modelname =='VGG19':
        model = models.vgg19(pretrained=True)
    elif modelname == 'ResNet152':
        model = models.resnet152(pretrained=True)
    elif modelname == 'GoogleNet':
        model = models.googlenet(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)

    model.eval()
    return model





def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


# ImageNet classes are often of the form `can_opener` or `Egyptian_cat`
# will use this method to properly format it so that we get
# `Can Opener` or `Egyptian Cat`
def format_class_name(class_name):
    class_name = class_name.replace('_', ' ')
    class_name = class_name.title()
    return class_name

def norm_ip(img):
    min = float(img.min())
    max = float(img.max())
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min + 1e-5)
    return img

def flow_st(images, flows,batch_size):

    # print(images.shape)
    H,W = images.size()[2:]
    

    # basic grid: tensor with shape (2, H, W) with value indicating the
    # pixel shift in the x-axis or y-axis dimension with respect to the
    # original images for the pixel (2, H, W) in the output images,
    # before applying the flow transforms
    grid_single = torch.stack(
                torch.meshgrid(torch.arange(0,H), torch.arange(0,W))
            ).float()


    if flows.shape[0] == batch_size:
        grid = grid_single
    else:
        grid = grid_single.repeat(batch_size, 1, 1, 1)#100,2,28,28

    images = images.permute(0,2,3,1) #100, 28,28,1


    
    grid_new = grid + flows
    # assert 0

    sampling_grid_x = torch.clamp(
        grid_new[:, 1], 0., (W - 1.)
            )
    sampling_grid_y = torch.clamp(
        grid_new[:, 0], 0., (H - 1.)
    )
    
    # now we need to interpolate

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a square around the point of interest
    x0 = torch.floor(sampling_grid_x).long()
    x1 = x0 + 1
    y0 = torch.floor(sampling_grid_y).long()
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate image boundaries
    # - 2 for x0 and y0 helps avoiding black borders
    # (forces to interpolate between different points)
    x0 = torch.clamp(x0, 0, W - 2)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 2)
    y1 = torch.clamp(y1, 0, H - 1)


    b =torch.arange(0, batch_size).view(batch_size, 1, 1).repeat(1, H, W)
 
    
    
    # assert 0 
    Ia = images[b, y0, x0].float()
    Ib = images[b, y1, x0].float()
    Ic = images[b, y0, x1].float()
    Id = images[b, y1, x1].float()


    x0 = x0.float()
    x1 = x1.float()
    y0 = y0.float()
    y1 = y1.float()

    wa = (x1 - sampling_grid_x) * (y1 - sampling_grid_y)
    wb = (x1 - sampling_grid_x) * (sampling_grid_y - y0)
    wc = (sampling_grid_x - x0) * (y1 - sampling_grid_y)
    wd = (sampling_grid_x - x0) * (sampling_grid_y - y0)

    # add dimension for addition
    wa = wa.unsqueeze(3)
    wb = wb.unsqueeze(3)
    wc = wc.unsqueeze(3)
    wd = wd.unsqueeze(3)

    # compute output
    perturbed_image = wa * Ia+ wb * Ib+ wc * Ic+wd * Id
 

    perturbed_image = perturbed_image.permute(0,3,1,2)

    return perturbed_image



