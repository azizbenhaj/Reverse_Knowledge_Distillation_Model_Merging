import os
import torch
import json
import glob
import collections
import random
import numpy as np
from tqdm import tqdm
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, Sampler

# Custom sampler for selecting a subset of indices
class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices  # Store the provided indices

    def __iter__(self):
        return (i for i in self.indices)  # Iterate over the indices

    def __len__(self):
        return len(self.indices)  # Return the number of indices

# Custom dataset class that extends ImageFolder to include image paths and random label flipping
class ImageFolderWithPaths(datasets.ImageFolder):
    def __init__(self, path, transform, flip_label_prob=0.0):
        super().__init__(path, transform)  # Initialize the parent class
        self.flip_label_prob = flip_label_prob  # Store the probability for label flipping
        if self.flip_label_prob > 0:
            print(f'Flipping labels with probability {self.flip_label_prob}')
            num_classes = len(self.classes)  # Get the number of classes
            # Randomly flip labels based on the specified probability
            for i in range(len(self.samples)):
                if random.random() < self.flip_label_prob:
                    new_label = random.randint(0, num_classes - 1)
                    self.samples[i] = (
                        self.samples[i][0],  # Keep the image path
                        new_label  # Assign a new label
                    )

    def __getitem__(self, index):
        # Retrieve the image and label from the parent class
        image, label = super(ImageFolderWithPaths, self).__getitem__(index)
        return {
            'images': image,  # Image tensor
            'labels': label,  # Label
            'image_paths': self.samples[index][0]  # Path to the image
        }

# Convert a batch to a dictionary format if it's not already in that format
def maybe_dictionarize(batch, labels_name='labels'):
    if isinstance(batch, dict):
        return batch  # Already a dictionary

    # Convert batch to dictionary based on its length
    if len(batch) == 2:
        batch = {'images': batch[0], labels_name: batch[1]}
    elif len(batch) == 3:
        batch = {'images': batch[0], labels_name: batch[1], 'metadata': batch[2]}
    else:
        raise ValueError(f'Unexpected number of elements: {len(batch)}')

    return batch

# Helper function to extract features using the image encoder
def get_features_helper(image_encoder, dataloader, device):
    all_data = collections.defaultdict(list)  # Store all extracted features

    image_encoder = image_encoder.to(device)  # Move the encoder to the specified device
    #image_encoder = torch.nn.DataParallel(image_encoder, device_ids=[x for x in range(torch.cuda.device_count())])
    image_encoder.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for batch in tqdm(dataloader):  # Iterate through the dataloader
            batch = maybe_dictionarize(batch)  # Ensure batch is a dictionary
            features = image_encoder(batch['images'].to("mps"))  # Extract features from images

            all_data['features'].append(features.cpu())  # Store features in all_data

            # Process other keys in the batch
            for key, val in batch.items():
                if key == 'images':
                    continue  # Skip images
                if hasattr(val, 'cpu'):
                    val = val.cpu()  # Move to CPU
                    all_data[key].append(val)  # Store in all_data
                else:
                    all_data[key].extend(val)  # Extend the list

    # Convert all lists in all_data to tensors
    for key, val in all_data.items():
        if torch.is_tensor(val[0]):
            all_data[key] = torch.cat(val).numpy()  # Concatenate tensors

    return all_data

# Function to get features from the dataset, either from cache or by computing them
def get_features(is_train, image_encoder, dataset, device):
    split = 'train' if is_train else 'val'  # Determine the split
    dname = type(dataset).__name__  # Get the dataset class name

    # Check for cached features
    if image_encoder.cache_dir is not None:
        cache_dir = f'{image_encoder.cache_dir}/{dname}/{split}'  # Define cache directory
        cached_files = glob.glob(f'{cache_dir}/*')  # List cached files

    if image_encoder.cache_dir is not None and len(cached_files) > 0:
        print(f'Getting features from {cache_dir}')  # Cache hit
        data = {}
        for cached_file in cached_files:
            name = os.path.splitext(os.path.basename(cached_file))[0]
            data[name] = torch.load(cached_file)  # Load cached features
    else:
        print(f'Did not find cached features at {cache_dir}. Building from scratch.')
        loader = dataset.train_loader if is_train else dataset.test_loader  # Select the appropriate dataloader
        data = get_features_helper(image_encoder, loader, device)  # Compute features
        if image_encoder.cache_dir is None:
            print('Not caching because no cache directory was passed.')
        else:
            os.makedirs(cache_dir, exist_ok=True)  # Create cache directory if it doesn't exist
            print(f'Caching data at {cache_dir}')  # Cache the computed features
            for name, val in data.items():
                torch.save(val, f'{cache_dir}/{name}.pt')

    return data  # Return extracted features

# Custom dataset class for storing extracted features
class FeatureDataset(Dataset):
    def __init__(self, is_train, image_encoder, dataset, device):
        self.data = get_features(is_train, image_encoder, dataset, device)  # Extract features on initialization

    def __len__(self):
        return len(self.data['features'])  # Return number of features

    def __getitem__(self, idx):
        # Get item at index idx and convert features to float tensor
        data = {k: v[idx] for k, v in self.data.items()}
        data['features'] = torch.from_numpy(data['features']).float()
        return data  # Return the item

# Function to get a dataloader for training or validation
def get_dataloader(dataset, is_train, args, image_encoder=None):
    if image_encoder is not None:
        # Create a FeatureDataset and DataLoader if an image encoder is provided
        feature_dataset = FeatureDataset(is_train, image_encoder, dataset, args.device)
        dataloader = DataLoader(feature_dataset, batch_size=args.batch_size, shuffle=is_train)
    else:
        # Use the dataset's default train or test loader
        dataloader = dataset.train_loader if is_train else dataset.test_loader
    return dataloader  # Return the DataLoader