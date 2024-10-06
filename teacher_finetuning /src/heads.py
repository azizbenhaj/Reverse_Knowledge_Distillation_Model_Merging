import os
import torch
from tqdm import tqdm

import open_clip

from all_datasets.templates import get_templates
from all_datasets.registry import get_dataset

from modeling import ClassificationHead, ImageEncoder


def build_classification_head(model, dataset_name, data_location, device):
    template = get_templates(dataset_name)

    # scaling factor applied to logits before applying the softmax function or another operation like normalization
    logit_scale = model.logit_scale 
    dataset = get_dataset(
        dataset_name,
        None,
        location=data_location
    )
    model.eval()
    model.to(device)

    print('Building classification head.')
    with torch.no_grad():  # Disable gradient computation for efficiency
        zeroshot_weights = []  # List to store the class embeddings
        for classname in tqdm(dataset.classnames):  # Iterate through class names in the dataset
            texts = []  # Initialize a list to hold generated text prompts for the class
            for t in template:  # For each template function
                texts.append(t(classname))  # Generate the text using the class name
            texts = open_clip.tokenize(texts).to(device)  # Tokenize the texts and move to device

            # Use the model to encode the tokenized texts into embeddings
            embeddings = model.encode_text(texts)
            # Normalize the embeddings to have unit length
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            # Average the embeddings for this class
            # construct a class prototype for each class, which is a representative embedding for that class based on multiple textual prompts
            # even though you have different text descriptions for the same class, you still end up with one output (embedding) per class
            embeddings = embeddings.mean(dim=0, keepdim=True)
            # Normalize the averaged embeddings
            embeddings /= embeddings.norm()

            # Append the processed embeddings to the zeroshot_weights list
            zeroshot_weights.append(embeddings)

        # Stack the class embeddings into a tensor and move to the device
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        # Transpose the weights to prepare for the classification head
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        # Scale the embeddings using the logit scale
        zeroshot_weights *= logit_scale.exp()

        # Convert the weights to float and transpose again for the classification head
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    # Create a classification head with normalized weights
    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head  # Return the constructed classification head

def get_classification_head(args, dataset):
    base_datase_name = dataset.split('Val')[0]
    filename = os.path.join(args.save, f'Classification_head_{args.model}_{base_datase_name}.pt')
    if os.path.exists(filename):
        print(f'Classification head for {args.model} on {dataset} exists at {filename}')
        return ClassificationHead.load(filename)
    print(f'Did not find classification head for {args.model} on {base_datase_name} at {filename}, building one from scratch.')
    model = ImageEncoder(args, keep_lang=True).model
    classification_head = build_classification_head(model, dataset, args.data_location, args.device)
    os.makedirs(args.save, exist_ok=True)
    classification_head.save(filename)
    return classification_head

def get_classification_head_kd(args, dataset):
    base_datase_name = dataset.split('Val')[0]
    filename = os.path.join(args.save, f'Classification_head_{args.model}_{base_datase_name}.pt')
    if os.path.exists(filename):
        print(f'Classification head for {args.model} on {dataset} exists at {filename}')
        return ClassificationHead.load(filename)
    print(f'Did not find classification head for {args.model} on {dataset} at {filename}, building one from scratch.')
    model = ImageEncoder(args, keep_lang=True).model
    classification_head = build_classification_head(model, dataset, args.data_location, args.device)
    os.makedirs(args.save, exist_ok=True)
    classification_head.save(filename)
    return classification_head