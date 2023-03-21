import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.io import read_image
import torchvision.transforms as transforms
from PIL import Image

# ----- image preprocessing -----
img_transforms = transforms.Compose([
    transforms.Resize((64, 64))
    ])

# Souce image here: this is the image we start with
src = read_image('dog.jpeg', mode=torchvision.io.ImageReadMode.RGB).float()
src = img_transforms(src)

# Target image here: this is the image we with to "merge" into
tgt = read_image('cat.jpeg', mode=torchvision.io.ImageReadMode.RGB).float()
tgt = img_transforms(tgt)


print(f"src: shape={src.shape} dtype={src.dtype}")
print(f"tgt: shape={tgt.shape} dtype={tgt.dtype}")


# ----- creating model -----
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(32),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
        )

    def __call__(self, x):
        out = self.model(x)

        output_img = transforms.ToPILImage()(out.type(torch.uint8))

        return output_img, out

# ----- initial set up -----
images = [ transforms.ToPILImage()(src.type(torch.uint8))]
epoch = 100000
model = Model()
criterion = nn.MSELoss()

lr = 1e-2

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ----- training -----
for i in range(epoch):
    optimizer.zero_grad()
    output_img, out = model(src)

    loss = criterion(out, tgt)

    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        print(f"epoch: {i}, loss: {loss}")
    if i % 100 == 0:
        images.append(output_img)

# ----- creating a gif file -----
def make_gif(images, filename):
    """creates a gif file given an array of images

    Args:
        images (list of images): array of images of convert into gif
        filename (string): a string for the filename
    """
    frames = images
    frame_one = frames[0]
    fname = filename + ".gif"
    frame_one.save(fname, format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)

make_gif(images, "result")