# pip imports
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import os

from matplotlib.patches import Circle

from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# Local Imports
from data import MIAS
from models import AFIM, UFCN



# Metric
import time



def main(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    

    # Retrieve model
    if args.model == 'afim':
        model = AFIM()

        # Import data
        metadata = os.path.join(args.data, 'test.txt')
        ds = MIAS(metadata, args.data)
        ds_loader = DataLoader(ds)

        model.load_state_dict(torch.load(args.model_path, weights_only=True))
    
        model.to(device)
        model.eval()
        


        pred_total = []
        label_total = []


        with torch.no_grad():
            correct = 0
            total = 0

            for i, data in enumerate(ds_loader):

                inputs = data[0].to(device)
                labels = data[1].to(device)

                outputs = model(inputs)
                _, predictions = torch.max(outputs, dim=1)

                total += labels.size(0)
                correct += (predictions == labels).sum().item()

                # get correct ds type for sklearn
                pred_total.extend(predictions.cpu().numpy())
                label_total.extend(labels.cpu().numpy())

        accuracy = correct / total
        print(f"Test accuracy: {accuracy*100:.2f}%")

        print(f'Classification Report:')
        print(classification_report(label_total, pred_total))
        
    elif args.model == 'ufcn':
        model = UFCN()

        # Import data
        metadata = os.path.join(args.data, 'test.txt')
        ds = MIAS(metadata, args.data, prior=True)
        ds_loader = DataLoader(ds)

        model.load_state_dict(torch.load(args.model_path, weights_only=True))
    
        model.to(device)
        model.eval()
        


        pred_total = []
        label_total = []


        with torch.no_grad():
            correct = 0
            total = 0

            for i, data in enumerate(ds_loader):

                inputs1 = data[0].to(device)
                inputs2 = data[1].to(device)
                labels = data[2].to(device)


                outputs, _, _ = model(inputs1, inputs2)

                logits_per_img = outputs.mean(dim=[2,3]).squeeze(1)
                predictions = (logits_per_img > 0).long()

                total += labels.size(0)
                correct = torch.sum(labels == predictions).item()

                # get correct ds type for sklearn
                pred_total.extend(predictions.cpu().numpy())
                label_total.extend(labels.cpu().numpy())

        accuracy = correct / total
        print(f"Test accuracy: {accuracy*100:.2f}%")

        print(f'Classification Report:')
        print(classification_report(label_total, pred_total))
    
    else:
        print('Unrecognized model - exiting...')
        quit(-1)

    



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str
    )

    parser.add_argument(
        '--model_path',
        type=str
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True
    )

    parser.add_argument(
        '--bs',
        type=int,
        required=False,
        default=4
    )
    args = parser.parse_args()
    main(args)