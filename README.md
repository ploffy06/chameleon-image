# chameleon-image
Given a source image, `chameleon-image` will "merge" into a target image. The result is stored as a `.gif` file.

This is done by training a model on a neural network, in which the model takes in the source image as the input. It learns via a MSE loss function where it computes the difference between the source and target image. Eventually, the model will learn to fit the source image to "become" the target image resulting in a sort of cool looking gif.

# Results
The following is the result of taking a dog as a source image and cat as the target image.

<p align="center" >
  <img src="result.gif" alt="animated" width="200" height="200"/>
</p>

# Running the script
If you wish to run the script with your own input images simply change the the first paramter in the following lines.

```
src = read_image('dog.jpeg', mode=torchvision.io.ImageReadMode.RGB).float()
...
tgt = read_image('cat.jpeg', mode=torchvision.io.ImageReadMode.RGB).float()
```

Please note that the `read_image` function can only take in either a `.jpeg` or `.png` file type. In addition, the `Resize()` transormation will skew that image to a given size. As I chose a size of `(64, 64)` it is reccomended that you either have square images or change the size variable.