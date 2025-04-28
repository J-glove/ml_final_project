# pip imports
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from matplotlib.patches import Circle

from PIL import Image
from torch.utils.data import DataLoader


# Local Imports
from data import MIAS
from models import AFIM, UFCN



# Metric
import time


# Debug for checking that the data is loaded correctly
def show_image(img):
    image = img[0].squeeze()
    if img[4] != None:
        cx, cy = img[4], img[5]      # center x,y
        radius = img[6]              # radius

        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')

        # draw ring
        ring = Circle((cx, cy),
                    radius,
                    edgecolor='red',    # ring color
                    facecolor='none',   # no fill
                    linewidth=2)
        ax.add_patch(ring)

        # draw center point
        ax.scatter([cx], [cy],
                c='red',
                s=50,               # marker size
                marker='x')
        plt.axis('off')  # optional, to hide axes
    
    plt.show()


def plot_data(data, name):
    # Takes in the loss or accuracy data and creates a plot and saves it
    print('')


def main():
    batch_size=8
    save_model=False

    # Check for CUDA hardware and use if available.
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


    # Load the dataset
    ds = MIAS('data_set/train/train.txt', 'data_set/train')
    ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # Declare the AFIM network and pass to the hardware accelerator
    afim = AFIM()
    afim.to(device)

    print(ds.__len__())

    #========== Debug ===========

    '''for i, data in enumerate(ds_loader):
        inputs,labels = data[0].to(device), data[1].to(device)
        print(f'Input shape: {inputs.shape}')
        print(f'Output shape: {afim(inputs).shape}')
        break

    quit(0)'''
    #============================




    # Train AFIM network
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(afim.parameters())

    epochs=1

    # Hold the loss and accuracy for plotting later
    afim_loss = []
    afim_acc = []
    start = time.time()
    for epoch in range(epochs):
        print(f'Epoch: {epoch+1}\n')

        afim.train(True)
        loss = 0.0
        accuracy = 0.0

        for i, data in enumerate(ds_loader):
            print(f"Batch #{i}")
            inputs = data[0]
            labels = data[1]

            optimizer.zero_grad()
            outputs = afim(inputs) 

            correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
            accuracy += correct / batch_size

            curr_loss = loss_function(outputs, labels)
            loss += curr_loss.item()
            if i % 10 == 0:
                print(f'Acc: {accuracy}')
                print(f'Loss: {loss}')
                #afim_acc.append([i, accuracy])
                afim_loss.append([i,loss])
            break
        
        end = time.time()
        run_time=end - start
        print(f"One epoch took {run_time:.2f} seconds")
        break


    plot_data(afim_acc, "AFIM: Accuracy (Test)")
    plot_data(afim_loss, "AFIM: Loss (Test)")

    if save_model:
        torch.save(afim.state_dict(), 'saved_models/afim.pt')


        

    #show_image(img)



    #afim = AFIM()





if __name__=='__main__':
    main()