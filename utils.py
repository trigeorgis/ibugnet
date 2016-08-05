import numpy as np

def caffe_preprocess(image):
    VGG_MEAN = np.array([102.9801, 115.9465, 122.7717])
    
    # RGB -> BGR
    pixels = image.pixels[[2, 1, 0]]
    # Subtract VGG training mean across all channels
    pixels = pixels - VGG_MEAN.reshape([3, 1, 1])
    pixels = pixels.astype(np.float32, copy=False)
    return pixels

def rescale_image(image, stride_width=64):
    # make sure smallest size is 600 pixels wide & dimensions are (k * stride_width) + 1
    height, width = image.shape

    # Taken from 'szross'
    scale_up = 625. / min(height, width)
    scale_cap = 961. / max(height, width)
    scale_up  = min(scale_up, scale_cap)
    new_height = stride_width * round((height * scale_up) / stride_width) + 1
    new_width = stride_width * round((width * scale_up) / stride_width) + 1
    image, tr = image.resize([new_height, new_width], return_transform=True)
    image.inverse_tr = tr
    return image