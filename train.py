# pip imports
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import os
import monai
from monai.metrics import DiceMetric
from monai import transforms as mt

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
    data = sorted(data, key=lambda pair: pair[0])
    x,y = zip(*data)

    plt.figure()
    plt.title(title)
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f'/content/plots/{f_name}.png')
    plt.close()

def main(args):
    batch_size=args.bs
    save_model=args.save

    data = args.data


    # Check for CUDA hardware and use if available.
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_meta = os.path.join(data, 'train.txt')
    

    if args.model=='afim':
        # Load the dataset
        ds = MIAS(train_meta, data)
        ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        # Declare the AFIM network and pass to the hardware accelerator
        afim = AFIM()
        afim.to(device)

        # Define metrics ds for graphing
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

                if i % 10 == 9: 
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
        
    elif args.model=='ufcn':
        ds = MIAS(train_meta, data, prior=True)
        ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        dice_metric = DiceMetric(include_background=True, reduction='mean')
        post_trans = mt.Compose([
            mt.Activations(sigmoid=True)
        ])

        
        ufcn = UFCN(activation = 'ReLU', threshold = 0.02)
        ufcn.to(device)

        loss_function = monai.losses.ssim_loss.SSIMLoss(spatial_dims=2)
        bce = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(ufcn.parameters(), 1e-2, weight_decay=1e-5)

        # start a typical PyTorch training
        epoch_loss_values = list()

        acc_plot = []
        loss_plot =[]

        accuracy = 0.0
        loss = 0.0
    
        for epoch in range(args.epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{args.epochs}")
            ufcn.train()
            epoch_loss = 0
            step = 0
            for i, data in enumerate(ds_loader):
                inputs1 = data[0].to(device)
                inputs2 = data[1].to(device)
                labels = data[2].to(device)
                step += 1
                data_range = inputs1.max().unsqueeze(0)
                outputs, _, labelLoss  = ufcn(inputs1, inputs2)
        
                print(data_range.shape)
                print(outputs.shape)
                print(inputs1.shape)
                l1 = loss_function(outputs, inputs1)

                term = 1
                layerLoss = 0
                for j in labelLoss:
                    preds = j.mean(dim=[1,2,3])

                    layerLoss = layerLoss + term * bce(preds.to(dtype = float), labels.to(dtype = float))
                    term -= 0.4

                l2 = layerLoss / len(labelLoss)

                cur_loss =  0.9 * l1 + 0.1 * l2
                loss += cur_loss
                #print(loss)

                optimizer.zero_grad()
                cur_loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                if i % 10 == 9: 
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
                #epoch_len = len(train_ds) // data_loader.batch_size
                #print(f"{step}, train_loss: {loss.item():.4f}")
            
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        print('Generating loss and accuracy plots')
        plot_data(acc_plot, 'UFCN Train Accuracy', 'Iterations', 'Accuracy', 
                'afim_acc')
        plot_data(loss_plot, 'UFCN Train Loss', 'Iterations', 'Loss', 
                'afim_loss')



    else:
        print('Invalid model')
        quit(-1)

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
    parser.add_argument(
        '--model',
        type=str
    )
    args = parser.parse_args()
    main(args)