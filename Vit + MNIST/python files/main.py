import os
from data_preparation import prepare_data, save_dataset, load_dataset, create_dataloader
from model_definition import get_vit_b_16_model, get_vit_b_32_model, save_model, load_pretrained_model
from training import train_model, train_knowledge_distillation
from evaluation import evaluate_model
from utils import show_batch
from torchvision import models
import torch
from torch import nn

# Script to perform data preparation, model training, evaluation

def main():

    # Install required modules
    os.system('pip install torch torchvision matplotlib numpy tqdm transformers datasets accelerate tensorboard evaluate --upgrade')

    # ------------------------------------------------------------------------------------------------------------------------------------

    # Prepare data
    proprietary_loader_train, proprietary_loader_test, open_subset_train, open_subset_test = prepare_data()

    # Show a batch of images
    dataiter = iter(proprietary_loader_train)
    images, labels = next(dataiter)
    print('Labels: ', labels)
    print('Batch shape: ', images.size())
    show_batch(images)

    # Get model and device
    teacher_model, device = get_vit_b_16_model(num_classes=10)
    teacher_model_0, device = get_vit_b_16_model(num_classes=10)

    # Train model
    print(device)
    train_losses = train_model(teacher_model, proprietary_loader_train, device, epochs=0)

    # Evaluate model
    accuracy_0 = evaluate_model(teacher_model_0, proprietary_loader_test, device)
    print(f"accuracy of the original model: {accuracy_0:.2f}%")
    accuracy = evaluate_model(teacher_model, proprietary_loader_test, device)
    print(f"accuracy of the new teacher model after pre training on MNIST: {accuracy:.2f}%")

    # Save the trained teacher moodel and the datasets used later for knowledge distillation
    save_model(teacher_model, "model/custom_MNIST_vit_b_16_model.pth")
    save_dataset(open_subset_train, 'data/open_unlabelled_subset_train.pkl')
    save_dataset(open_subset_test, 'data/open_unlabelled_subset_test.pkl')

    print("done with training teacher model on MNIST")

    # ------------------------------------------------------------------------------------------------------------------------------------

    # Load the training open dataset
    open_unlabelled_subset_train = load_dataset('data/open_unlabelled_subset_train.pkl')
    open_unlabelled_subset_test = load_dataset('data/open_unlabelled_subset_test.pkl')

    # Create DataLoader objects for the open data
    open_unlabelled_loader_train = create_dataloader(open_unlabelled_subset_train)
    open_unlabelled_loader_test = create_dataloader(open_unlabelled_subset_test)

    # Show a batch of images
    dataiter = iter(open_unlabelled_loader_train)
    images, labels = next(dataiter)
    print('Labels: ', labels)
    print('Batch shape: ', images.size())
    show_batch(images)

    # Load teacher model already trained on MNIST
    teacher_model = load_pretrained_model(models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1), 
                               "model/custom_MNIST_vit_b_16_model.pth", 10, torch.device("mps"))
    
    # Load 3 student models (1st remains unchanged for comparison, 2nd trained using only DL, 3rd trained using CE+DL)

    student_model_0, device = get_vit_b_32_model(num_classes=10)
    student_model_kd, device = get_vit_b_32_model(num_classes=10)
    student_model_ce_kd, device = get_vit_b_32_model(num_classes=10)

    # Train knowledge distillation using only DL
    T=2
    learning_rate=0.001
    epochs=0  #5 #10
    lambda_param = 0.5
    device = torch.device("mps")
    train_knowledge_distillation(teacher=teacher_model, student=student_model_ce_kd, train_loader=open_unlabelled_loader_train,
                                epochs=epochs, learning_rate=learning_rate, T=T, lambda_param=lambda_param, device=device)

    # Train knowledge distillation using CE+DL
    lambda_param = 1
    train_knowledge_distillation(teacher=teacher_model, student=student_model_kd, train_loader=open_unlabelled_loader_train,
                                epochs=epochs, learning_rate=learning_rate, T=T, lambda_param=lambda_param, device=device)

    # Evaluating all models' performances
    test_accuracy_teacher = evaluate_model(teacher_model, open_unlabelled_loader_test, device=device)
    test_accuracy_student_before_kd = evaluate_model(student_model_0, open_unlabelled_loader_test, device=device)
    test_accuracy_student_after_kd = evaluate_model(student_model_kd, open_unlabelled_loader_test, device=device)
    test_accuracy_student_after_ce_kd = evaluate_model(student_model_ce_kd, open_unlabelled_loader_test, device=device)

    # Compare the student test accuracy with and without the teacher, after distillation
    print(f"Teacher accuracy: {test_accuracy_teacher:.2f}%")
    print(f"Student accuracy without teacher: {test_accuracy_student_before_kd:.2f}%")
    print(f"Student accuracy with KD: {test_accuracy_student_after_kd:.2f}%")
    print(f"Student accuracy with CE + KD: {test_accuracy_student_after_ce_kd:.2f}%")


    print("done with reverse knowledge ditillation")

    # ------------------------------------------------------------------------------------------------------------------------------------
    
if __name__ == "__main__":
    main()
