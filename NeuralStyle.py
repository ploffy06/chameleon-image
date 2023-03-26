import torch
import torch.nn as nn
import numpy as np
from torchvision.io import read_image
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image

# ----- helper functions -----
device = 'gpu' if torch.cuda.is_available() else 'cpu'
def image_loader(image_name):
    """_summary_

    Args:
        image_name (string): file location and name of image

    Returns:
        tensor: the vector repr. of image with dtype=float32 and size=[1, 3, 256, 256]
    """
    loader = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()])

    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def save_best_image(images):
    """saves the "best" (most "colourful") image as a jpeg

    Args:
        images (3 x 256 x 256 tensor): vector representation of image in type uint8
    """
    print("saving best image...")
    max = 0
    best_image = images[0]
    for image in images:
        if image.abs().sum().item() > max:
            max = image.abs().sum().item()
            best_image = image

    img = transforms.ToPILImage()(best_image)
    img.save(f"results/{content_img_name}x{style_img_name}_512.jpeg")

def make_gif(images):
    """creates a gif file given an array of images

    Args:
        images (list of images): array of images of convert into gif
    """
    frames = [transforms.ToPILImage()(image) for image in images]
    frame_one = frames[0]
    fname = f"results/{content_img_name}x{style_img_name}_512.gif"
    frame_one.save(fname, format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)

def get_gram(input):
    """returns the gram matrix

    Args:
        input (tensor): input tensor

    Returns:
        tensor: gram matrix of input tensor
    """
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)

    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

def get_input_optimizer(input_img):
    optimizer = torch.optim.LBFGS([input_img])
    return optimizer

# ----- image preprocessing -----
content_img_name = str(input("content file name: "))
style_img_name = 'van'
content = image_loader(f'{content_img_name}.jpeg')
style = image_loader(f'{style_img_name}.jpeg')

print(f"{content_img_name} content image: shape={content.shape} dtype={content.dtype}")
print(f"Van Gough style image: shape={style.shape} dtype={style.dtype}")


# ----- Content loss function -----
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

# ----- Style loss function -----
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = get_gram(target_feature).detach()

    def forward(self, input):
        G = get_gram(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# ----- vgg model -----
cnn = models.vgg19().features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

# ----- getting content and style losses -----
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

# ----- running style transfer -----
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=2500,
                       style_weight=1000000, content_weight=1):
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    images = []
    while run[0] <= num_steps:
        curr_img = input_img[0].type(torch.uint8).clone()
        images.append(curr_img)
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)
    return input_img, images


input = content.clone()
output, images = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content, style, input)

save_best_image(images)
make_gif(images)
