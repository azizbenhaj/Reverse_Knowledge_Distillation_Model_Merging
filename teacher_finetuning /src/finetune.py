import os
import time
import torch
from tqdm import tqdm

import wandb

from args import parse_arguments
from eval import evaluate
from modeling import ImageEncoder, ImageClassifier, MultiHeadImageClassifier
from utils import cosine_lr, LabelSmoothing
from heads import get_classification_head
from all_datasets.common import get_dataloader, maybe_dictionarize
from all_datasets.registry import get_dataset
#import datasets as datasets



def finetune(args):
    print("Starting finetuning teacher...")
    train_dataset = args.train_dataset
    base_dataset_name = train_dataset.split('Val')[0]
    ckpdir = os.path.join(args.save, base_dataset_name) #where to save models

    # Check if checkpoints already exist args.model
    zs_path = os.path.join(args.save, train_dataset, f'zeroshot_{args.model}_{base_dataset_name}.pt')  
    ft_path = os.path.join(args.save, train_dataset, f'finetuned_{args.model}_{base_dataset_name}_{args.epochs}.pt')
    if os.path.exists(zs_path) and os.path.exists(ft_path):
        print(f'Skipping fine-tuning because {ft_path} exists.')
        return zs_path, ft_path

    assert train_dataset is not None, "Please provide a training dataset."

    print("Loading or Building image encoder ?")
    if args.load is not None and args.load.endswith('pt'):
        image_encoder = ImageEncoder.load(args.load)
    else:
        print('Building image encoder.')
        image_encoder = ImageEncoder(args, keep_lang=False)
    
    print("Building classification head")
    classification_head = get_classification_head(args, train_dataset)

    print("Building ImageClassifier (ImageEncoder + classification head)")
    model = ImageClassifier(image_encoder, classification_head)

    # prevent the weights of the classification head from being updated during the initial stages of training
    # This is especially important since the classification head is newly added and initialized from scratch,
    # as updating it prematurely can destabilize training.
    # The common strategy here is to freeze the classification head initially while you fine-tune the image encoder 
    # (the backbone), especially if you want to transfer a pre-trained image encoder to a new task.
	# After fine-tuning the image encoder for a while, you can unfreeze the classification head and allow both the 
    # image encoder and the classification head to be trained together. This helps in gradually adapting both components 
    # to the new task without large fluctuations in the classification head.
    print("Freezing head ...")
    model.freeze_head()

    preprocess_fn = model.train_preprocess
    # how often, in terms of steps/iterations, the training status is printed during the training loop
    print_every = 2 #100

    print("getting dataset ...")
    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )

    num_batches = len(dataset.train_loader)
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
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    print("saving zero-shot model ...")
    if args.save is not None:
        os.makedirs(ckpdir, exist_ok=True)
        model_path = os.path.join(ckpdir, f'image_encoder_zeroshot_{args.model}_{base_dataset_name}.pt')
        model.image_encoder.save(model_path)

    print("training ...")
    for epoch in tqdm(range(args.epochs), desc='Training Epochs'):
        model = model.to("mps")
        model.train()
        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=None)
        for i, batch in enumerate(data_loader):
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

            # Forward pass
            logits = model(inputs)
            # Calculate loss
            loss = loss_fn(logits, labels)

            loss.backward()
            # prevent the gradients from becoming too large during the backpropagation step
            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            batch_time = time.time() - start_time

            #if step % print_every == 0:
            percent_complete = 100 * i / len(data_loader)
            print(
                f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
            )
            # Log training loss to wandb
            wandb.log({"train_loss": loss.item(), "epoch": epoch, "step": step})
    
    print("saving finetuned model ...")
    if args.save is not None: 
        ft_path = os.path.join(ckpdir, f'image_encoder_finetuned{args.model}_{base_dataset_name}_{args.epochs}.pt')
        image_encoder.save(ft_path)
    
    # Evaluate
    print("evaluating ...")
    image_encoder = model.image_encoder
    num_batches = len(dataset.test_loader)
    print(num_batches)
    data_loader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    acc, info = evaluate(image_encoder, data_loader, args)
    
    wandb.log({"test_accuracy": acc})

if __name__ == '__main__':
    data_location = 'datasets_directory' # The root directory for the datasets
    models = ['ViT-B-16'] #['ViT-B-32', 'ViT-B-16', 'ViT-L-14']
    datasets = ['STL10']#, 'MNIST'] #['Cars', 'DTD', 'EuroSAT', 'GTSRB', 'MNIST', 'RESISC45', 'SUN397', 'SVHN']
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
            
            print('='*100)
            print(f'Finetuning {model} on {dataset}')
            print('='*100)
            args = parse_arguments()
            args.lr = 1e-5
            args.epochs = epochs[dataset]
            args.data_location = data_location
            args.train_dataset = dataset + 'Val'
            args.eval_datasets = dataset
            args.batch_size = 64
            args.model = model
            args.pretrained = 'laion400m_e32'
            args.save = f'checkpoints/{model}'
            args.device = "mps"
            args.results_db = f'results/{model}/{dataset}'
            args.labels_name = labels_name[dataset]

            wandb.init(
                # set the wandb project where this run will be logged
                project=f"finetuning_{model}_{dataset}",

                # track hyperparameters and run metadata
                config={
                "learning_rate": args.lr,
                "architecture": model,
                "dataset": dataset,
                "epochs": epochs[dataset],
                "label_smoothing": args.ls
                }
            )

            finetune(args)