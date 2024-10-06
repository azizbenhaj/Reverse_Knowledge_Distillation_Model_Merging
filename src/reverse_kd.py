import os
import time
import torch
from tqdm import tqdm

import wandb

from args import parse_arguments
from modeling import ImageEncoder, ImageClassifier, MultiHeadImageClassifier
from utils import cosine_lr, LabelSmoothing
from heads import get_classification_head, get_classification_head_kd
from all_datasets.common import maybe_dictionarize
#import datasets as datasets
import torch.nn.functional as F
from torch import nn

def rkd(args):
    print("Starting knowledge distillation from teacher to student...")
    train_dataset = args.train_dataset
    base_dataset_name = train_dataset.split('Val')[0]
    ckpdir = os.path.join(args.save, base_dataset_name) #where to save models

    # Check if checkpoints already exist
    zs_path = os.path.join(ckpdir, f'image_encoder_zeroshot_kd_{args.model}_{base_dataset_name}.pt')  
    ft_path = os.path.join(ckpdir, f'image_encoder_distillation_{args.model}_{base_dataset_name}_{args.epochs}.pt')

    if os.path.exists(zs_path) and os.path.exists(ft_path):
        print(f'Skipping knowledge distillation because {ft_path} exists.')
        return zs_path, ft_path
    assert train_dataset is not None, "Please provide a training dataset."

    print("Loading teacher model...")
    teacher_image_encoder = ImageEncoder.load_model(os.path.join(ckpdir, f'image_encoder_zeroshot_{args.model}_{base_dataset_name}.pt'))
    teacher_classification_head = get_classification_head(args, train_dataset)
    teacher_model = ImageClassifier(teacher_image_encoder, teacher_classification_head)
    print("teacher model loaded successfully")
    
    print("Initializing student image encoder...")
    student_image_encoder = ImageEncoder(args, keep_lang=False)
    classification_head = get_classification_head_kd(args, train_dataset)
    student_model = ImageClassifier(student_image_encoder, classification_head)
    print("student model initialised successfully")

    print("getting datasets and dataloaders...")
    split_save_dir = os.path.join(args.data_location, 'subsets', base_dataset_name)
    train_kd_data = torch.load(os.path.join(split_save_dir, 'distillation_train_set.pt'))
    test_kd_data = torch.load(os.path.join(split_save_dir, 'distillation_test_set.pt'))
    train_kd_loader = torch.utils.data.DataLoader(train_kd_data, batch_size=args.batch_size, shuffle=True, num_workers=16)
    test_kd_loader = torch.utils.data.DataLoader( test_kd_data, batch_size=args.batch_size, shuffle=True, num_workers=16)

    num_batches = len(train_kd_loader)
    print(num_batches)
    device = 'mps'
    print('Using device', device)
    
    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
        print("using LabelSmoothing ...")
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        print("using CrossEntropyLoss ...")

    print("defining optimizer and scheduler ...")
    params = [p for p in student_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    print("saving zero-shot distillation model ...")
    if args.save is not None:
        os.makedirs(ckpdir, exist_ok=True)
        model_path = os.path.join(ckpdir, f'image_encoder_zeroshot_kd_{args.model}_{base_dataset_name}.pt')
        student_model.image_encoder.save(model_path)


    print("training ...")
    T=2
    lambda_param = 1 
    print_every = 100
    running_loss = 0

    for epoch in tqdm(range(args.epochs), desc='Training Epochs for KD'):
        teacher_model = teacher_model.to("mps")
        teacher_model.eval()
        student_model = student_model.to("mps")
        student_model.train()

        for i, batch in enumerate(train_kd_loader):
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch, args.labels_name)
            inputs = batch['images'].to('mps')
            labels = batch[args.labels_name]

            if args.labels_name == 'car_models':
                labels = labels.squeeze()
                labels = labels.type(torch.int64)
 
            labels = labels.to('mps')
            data_time = time.time() - start_time
            
            # Forward pass with the teacher model (gradients not saved)
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)
            # Forward pass with the student model
            student_logits = student_model(inputs)

            soft_teacher = F.softmax(teacher_logits / T, dim=-1)
            soft_student = F.log_softmax(student_logits / T, dim=-1)

            distillation_fct = nn.KLDivLoss(reduction="batchmean")
            distillation_loss = distillation_fct(soft_student, soft_teacher) * (T ** 2)

            ce_loss = loss_fn(student_logits, labels)

            # Weighted sum of the two losses
            final_loss = (1. - lambda_param) * ce_loss + lambda_param * distillation_loss

            final_loss.backward()

            # prevent the gradients from becoming too large during the backpropagation step
            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            batch_time = time.time() - start_time

            running_loss += final_loss.item()

            #if step % print_every == 0:
            percent_complete = 100 * i / len(train_kd_loader)
            print(
                f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(train_kd_loader)}]\t"
                f"Loss: {final_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
            )
            # Log training loss to wandb
            wandb.log({"train_loss": final_loss.item(), "epoch": epoch, "step": step})

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {running_loss / len(train_kd_loader)}")

    
    print("saving distilled model ...")
    if args.save is not None: 
        ft_path = os.path.join(ckpdir, f'image_encoder_distillation_{args.model}_{base_dataset_name}_{args.epochs}.pt')
        student_model.image_encoder.save(ft_path)


    print("testing student model after distillation...")
    student_model.to(device)
    student_model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for i, batch in enumerate(test_kd_loader):
            batch = maybe_dictionarize(batch, args.labels_name)
            inputs = batch['images'].to('mps')
            labels = batch[args.labels_name]

            if args.labels_name == 'car_models':
                labels = labels.squeeze()
                labels = labels.type(torch.int64)
 
            labels = labels.to('mps')
            
            outputs = student_model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy for student model after distillation: {accuracy:.2f}%")

    wandb.log({"test_accuracy": accuracy})


if __name__ == '__main__':
    data_location = 'datasets_directory' # The root directory for the datasets
    models = ['ViT-B-16'] #['ViT-B-32', 'ViT-B-16', 'ViT-L-14']
    student_models = ['ViT-B-32']
    datasets = ['STL10'] #['Cars', 'DTD', 'EuroSAT', 'GTSRB', 'MNIST', 'RESISC45', 'SUN397', 'SVHN']
    epochs = {
        'Cars': 1, #35, #still have code issues with Cars: TODO later on
        'DTD': 1, #76,
        'EuroSAT': 12,
        'GTSRB': 11,
        'MNIST': 1, #5,
        'RESISC45': 15,
        'SUN397': 14,
        'SVHN': 1, #4,
        'ImageNet': 4,
        'CIFAR100': 1, #I have to find the best number of epochs
        'CIFAR10': 1, #I have to find the best number of epochs
        'STL10': 3, #I have to find the best number of epochs
    }

    labels_name = {
        'Cars': "car_models",
        'MNIST': "labels",
        'DTD': "labels",
        'CIFAR100': "labels",
        'CIFAR10': "labels",
        'SVHN': "labels",
        'STL10': "labels",
    }

    for model in models:
        for dataset in datasets:
            for student in student_models:
                print('='*100)
                print(f'Distilling {model} on {student} using {dataset}')
                print('='*100)
                args = parse_arguments()
                args.lr = 1e-5
                args.epochs = epochs[dataset]
                args.data_location = data_location
                args.train_dataset = dataset + 'Val'
                args.student_model = student
                args.batch_size = 64
                args.model = model
                args.pretrained = 'laion400m_e32'
                args.save = f'checkpoints/{model}'
                args.device = "mps"
                args.results_db = f'results/{model}'
                args.labels_name = labels_name[dataset]

                wandb.init(
                    # set the wandb project where this run will be logged
                    project=f"knowledge_distillation_from_{model}_to_{student}_on{dataset}",

                    # track hyperparameters and run metadata
                    config={
                    "learning_rate": args.lr,
                    "architecture_teacher": model,
                    "architecture_student": student,
                    "dataset": dataset,
                    "epochs": epochs[dataset],
                    "label_smoothing": args.ls
                    }
                )
                
                rkd(args)
