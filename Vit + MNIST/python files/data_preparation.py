import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import pickle

# Code related to data loading, splitting, and transformation

def prepare_data():
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
    ])

    # Download and load the MNIST training dataset
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    # Download and load the MNIST test dataset
    testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)

    # Split the trainset and testset into proprietary labeled and open unlabelled data
    proprietary_size_train = len(trainset) // 2
    total_size_train = len(trainset)
    proprietary_size_test = len(testset) // 2
    total_size_test = len(testset)

    # Create Proprietary labeled data indices and Open unlabelled data indices
    indices_train = torch.randperm(total_size_train)
    proprietary_indices_train = indices_train[:proprietary_size_train]
    open_indices_train = indices_train[proprietary_size_train:]

    indices_test = torch.randperm(total_size_test)
    proprietary_indices_test = indices_test[:proprietary_size_test]
    open_indices_test = indices_test[proprietary_size_test:]

    # Create proprietary labeled subset and open unlabelled subset
    proprietary_subset_train = Subset(trainset, proprietary_indices_train)
    open_subset_train = Subset(trainset, open_indices_train)
    proprietary_subset_test = Subset(testset, proprietary_indices_test)
    open_subset_test = Subset(testset, open_indices_test)

    # Create DataLoader objects for both proprietary and open data
    proprietary_loader_train = DataLoader(proprietary_subset_train, batch_size=64, shuffle=True)
    #open_unlabelled_loader_train = DataLoader(open_subset_train, batch_size=64, shuffle=True)
    proprietary_loader_test = DataLoader(proprietary_subset_test, batch_size=64, shuffle=True)
    #open_unlabelled_loader_test = DataLoader(open_subset_test, batch_size=64, shuffle=True)

    print(f"Proprietary labeled train data size: {len(proprietary_subset_train)}")
    print(f"Proprietary labeled test data size: {len(proprietary_subset_test)}")
    print(f"Open unlabelled train data size:  {len(open_indices_train)}")
    print(f"Open unlabelled test data size: {len(open_indices_test)}")

    print(f"Proprietary labeled train data size: {len(proprietary_loader_train)}")
    #print(f"Open unlabelled train data size: {len(open_unlabelled_loader_train)}")
    print(f"Proprietary labeled test data size: {len(proprietary_loader_test)}")
    #print(f"Open unlabelled train data size: {len(open_unlabelled_loader_test)}")

    return proprietary_loader_train, proprietary_loader_test, open_subset_train, open_subset_test


# Save dataset using pickle
def save_dataset(dataset, path):
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)
    
# Load dataset using pickle
def load_dataset(path):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

# Create DataLoader objects for the DataSet
def create_dataloader(dataset, batch_size=64, shuffle=True):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print(f"Open dataloader size: {len(dataloader)}")
    return dataloader