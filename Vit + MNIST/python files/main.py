import os
from data_preparation import prepare_data
from model_definition import get_vit_b_16_model
from training import train_model
from evaluation import evaluate_model
from utils import show_batch, save_model, save_dataset

# Script to perform data preparation, model training, evaluation

def main():

    # Install required modules
    os.system('pip install torch torchvision matplotlib numpy tqdm')
    # Prepare data
    proprietary_loader_train, proprietary_loader_test, open_subset_train, open_subset_test = prepare_data()

    # Show a batch of images
    dataiter = iter(proprietary_loader_train)
    images, labels = next(dataiter)
    show_batch(images)
    print('Labels: ', labels)
    print('Batch shape: ', images.size())

    # Get model and device
    model, device = get_vit_b_16_model(num_classes=10)
    model_0, device = get_vit_b_16_model(num_classes=10)

    # Train model
    print(device)
    train_losses = train_model(model, proprietary_loader_train, device, epochs=0)

    # Evaluate model
    print("accuracy of the original model:")
    accuracy_0 = evaluate_model(model_0, proprietary_loader_test, device)
    print("accuracy of the new teacher model after pre training on MNIST:")
    accuracy = evaluate_model(model, proprietary_loader_test, device)

    save_model(model, "model/custom_MNIST_vit_b_16_model.pth")

    save_dataset(open_subset_train, 'data/open_unlabelled_subset_train.pkl')
    save_dataset(open_subset_test, 'data/open_unlabelled_subset_test.pkl')
    
if __name__ == "__main__":
    main()
