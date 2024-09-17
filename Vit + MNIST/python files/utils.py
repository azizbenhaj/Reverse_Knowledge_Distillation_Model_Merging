import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
import pickle

# Helper functions

def show_batch(images):
    im = torchvision.utils.make_grid(images)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    plt.show()

# Save the model's state dictionary
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Save dataset using pickle
def save_dataset(dataset, path):
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)