import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
from torch import nn, optim
from torchvision import models

# Helper functions

def show_batch(images):
    im = torchvision.utils.make_grid(images)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    plt.show()








