import os
import json
import tqdm

import torch
import numpy as np

import utils
from all_datasets.common import get_dataloader, maybe_dictionarize
from heads import get_classification_head
from modeling import ImageClassifier

from all_datasets.registry import get_dataset


def eval_dataset(image_encoder, dataset_name, data_loader, args):
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)

    model.eval()

    '''dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)'''
    device = args.device

    with torch.no_grad():
        # top1: gives the overall accuracy of the model as a fraction of correctly predicted samples over total samples.
        # correct: counts how many predictions were correct.
	    # n: counts how many samples were evaluated.
        top1, correct, n = 0., 0., 0.

        for i, data in enumerate(tqdm.tqdm(data_loader)):

            data = maybe_dictionarize(data, args.labels_name)
            x = data['images'].to(device)
            y = data[args.labels_name]
            
            if args.labels_name == 'car_models':
                y = y.squeeze()
                y = y.type(torch.int64)
            
            y = y.to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()
            
            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating {args.model} on {dataset_name}. Accuracy: {100*top1:.2f}%')
    
    return metrics

def evaluate(image_encoder, data_loader, args):
    if args.eval_datasets is None:
        return
    
    # returns the __dict__ attribute of the given object.
    info = vars(args)
    dataset_name = args.eval_datasets
    print(f'Evaluating {args.model} on', dataset_name)

    results = eval_dataset(image_encoder, dataset_name, data_loader, args)

    if 'top1' in results:
        print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
    for key, val in results.items():
        if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
            print(f"{dataset_name} {key}: {val:.4f}")
        # organizes information in info dictionary by associating metrics with specific datasets
        info[dataset_name + ':' + key] = val

    if args.results_db is not None:
        dirname = os.path.dirname(args.results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.results_db, 'a+') as f:
            f.write(json.dumps(info) + '\n')
        print(f'Results saved to {args.results_db}.')
    else:
        print('Results not saved (to do so, use --results_db to specify a path).')

    return results['top1'], info