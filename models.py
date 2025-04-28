import torch
import torch.nn as nn
import torch.nn.functional as F



#=======================================================
#------ Attention-based Feature Integration Model ------
#=======================================================
class AFIM(nn.Module):
    def __init__(self):
        super().__init__()


        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.afim1 = afim_layer(in_channels=256, reduction=1, kernel=7)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.afim2 = afim_layer(in_channels=256, reduction=1, kernel=7)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.afim3 = afim_layer(in_channels=256, reduction=1, kernel=7)

        self.pool = nn.MaxPool2d(2,2)
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 8)
        )

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
        x = self.encode(x) # Shape: [8, 256, 1024, 1024]

        x = F.relu(self.afim1(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.afim2(x))
        x = F.relu(self.conv5(x))

        x = self.pool(x)

        x = self.final(x)

        return x
        



class afim_layer(nn.Module):
    # A custom layer for the CNN defined in AFIM to perform the AM step in the paper.
    # Using single view rather than bilateral
    #
    # Implements Single view Channel attention and Spatial attention outlined here: https://arxiv.org/pdf/1807.06521v2
    
    def __init__(self, in_channels, reduction=1, kernel=7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel,
                      padding=kernel//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_ca = F.adaptive_avg_pool2d(x,1)
        max_ca = F.adaptive_max_pool2d(x,1)

        # channel attention
        ca = self.sigmoid(self.mlp(avg_ca)+self.mlp(max_ca))
        x = x * ca

        # spatial attention
        avg_sa = x.mean(dim=1, keepdim=True)
        max_sa = x.max(dim=1, keepdim=True)[0]
        sa = self.sa(torch.cat([avg_sa,max_sa], dim=1))
        x = x * sa

        return x

        



#======================================================
#------ Unsupervised Feature Correlation Network ------
#======================================================
class UFCN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        print('')

