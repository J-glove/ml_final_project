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

    # Use GPU if possible
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Import data
    metadata = os.path.join(args.data, 'info.txt')
    ds = MIAS(metadata, args.data)
    ds_loader = DataLoader(ds)
    
    
    # Retrieve model
    model = torch.load(args.model, weights_only=False)
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



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
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