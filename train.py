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


# Local Imports
from data import MIAS
from models import AFIM, UFCN



# Metric
import time
def plot_data(data, title, x_label, y_label, f_name):
    # data ds should have entries like [x,y] where x is epoch + iteration/batchsize
    # and y is whatever is measured (loss, accuracy)
    plt.figure()
    plt.title(title)
    plt.plot(data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f'plots/{f_name}.png')

def main(args):
    batch_size=args.bs
    save_model=args.save

    data = args.data


    # Check for CUDA hardware and use if available.
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_meta = os.path.join(data, 'train.txt')
    # Load the dataset
    ds = MIAS(train_meta, data)
    ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # Declare the AFIM network and pass to the hardware accelerator
    afim = AFIM()
    afim.to(device)


    #========== Debug ===========

    '''for i, data in enumerate(ds_loader):
        inputs,labels = data[0].to(device), data[1].to(device)
        print(f'Input shape: {inputs.shape}')
        print(f'Output shape: {afim(inputs).shape}')
        break

    quit(0)'''
    #============================


    # Define metrics ds' for graphing
    acc_plot = []
    loss_plot = []

    # Train AFIM network
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(afim.parameters())

    epochs= args.epochs
    start = time.time()
    for epoch in range(epochs):
        print(f'Epoch: {epoch+1}\n')

        afim.train(True)
        loss = 0.0
        accuracy = 0.0

        for i, data in enumerate(ds_loader):
            
            torch.cuda.empty_cache()
            inputs = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()
            outputs = afim(inputs) 

            correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
            accuracy += correct / batch_size

            curr_loss = loss_function(outputs, labels)
            loss += curr_loss.item()

            curr_loss.backward()
            optimizer.step()

            if i % 9 == 0: #Offset by 1 for 0 index of i
                avg_loss = loss / 10
                avg_acc = (accuracy / 10) * 100
                print('Batch {0}, Loss: {1:.3f}, Accuracy: {2:.1f}%'.format(i+1,
                                                          avg_loss,
                                                          avg_acc))
                

                x = epoch + (i / batch_size)
                loss_plot.append([x,avg_loss])
                acc_plot.append([x, avg_acc])


                accuracy=0.0
                loss=0.0
        
        end = time.time()
        run_time=end - start
        print(f"One epoch took {run_time:.2f} seconds")

    print('Generating loss and accuracy plots')
    plot_data(acc_plot, 'AFIM Train Accuracy', 'Iterations', 'Accuracy', 
              'afim_acc')
    plot_data(loss_plot, 'AFIM Train Loss', 'Iterations', 'Loss', 
              'afim_loss')

    if save_model:
        print(f'Saving model to {args.out}...')
        torch.save(afim.state_dict(), args.out)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        required=True
    )

    parser.add_argument(
        '--bs',
        type=int,
        required=False,
        default=8
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=10

    )
    parser.add_argument(
        '--out',
        type=str,
        required=False
    )
    parser.add_argument(
        '--save',
        required=False,
        default=False,
        type=bool
    )
    args = parser.parse_args()
    main(args)