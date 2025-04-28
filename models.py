import torch
import torch.nn as nn
import torch.nn.functional as F
import layers



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
    # Model pulled from https://github.com/NabaviLab/ufcn/tree/main
    def __init__(self, img_ch = 1, output_ch = 1, block1Ch = 64, block2Ch = 128, block3Ch = 256, block4Ch = 512, block5Ch = 1024, activation = 'Relu', threshold = 0.0):
        super(UFCN, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.Conv1 = layers.conv_block(inputD = img_ch, outputD = block1Ch, activation = activation, threshold = threshold)
        self.Conv2 = layers.conv_block(inputD = block1Ch, outputD = block2Ch, activation = activation, threshold = threshold)
        self.Conv3 = layers.conv_block(inputD = block2Ch, outputD = block3Ch, activation = activation, threshold = threshold)
        self.Conv4 = layers.conv_block(inputD = block3Ch, outputD = block4Ch, activation = activation, threshold = threshold)
        self.Conv5 = layers.conv_block(inputD = block4Ch, outputD = block5Ch, activation = activation, threshold = threshold)

        self.up5 = layers.up_conv(inputD = block5Ch, outputD = block4Ch, activation = activation, threshold = threshold)
        self.att5 = layers.Attention_block(in_channels = block4Ch, gating_channels = block4Ch, inter_channels = 128)
        self.Up_conv5 = layers.conv_block(inputD = block5Ch, outputD = block4Ch, activation = activation, threshold = threshold)

        self.up4 = layers.up_conv(inputD = block4Ch, outputD = block3Ch, activation = activation, threshold = threshold)
        self.att4 = layers.Attention_block(in_channels = block3Ch, gating_channels = block3Ch, inter_channels = 256)
        self.Up_conv4 = layers.conv_block(inputD = block4Ch, outputD = block3Ch, activation = activation, threshold = threshold)      

        self.up3 = layers.up_conv(inputD = block3Ch, outputD = block2Ch, activation = activation, threshold = threshold)
        self.att3 = layers.Attention_block(in_channels = block2Ch, gating_channels = block2Ch, inter_channels = 512)
        self.Up_conv3 = layers.conv_block(inputD = block3Ch, outputD = block2Ch, activation = activation, threshold = threshold)
    
        self.up2 = layers.up_conv(inputD = block2Ch, outputD = block1Ch, activation = activation, threshold = threshold)
        self.att2 = layers.Attention_block(in_channels = block1Ch, gating_channels = block1Ch, inter_channels = 1024)
        self.Up_conv2 = layers.conv_block(inputD = block2Ch, outputD = block1Ch, activation = activation, threshold = threshold)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size = 1, stride = 1, padding = 0)

        self.agr = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()
       # self.mask = nn.Threshold(0.5,0)
        
    def forward(self, x1, x2):
   
        x1_1 = self.Conv1(x1)
        x2_1 = self.Conv1(x2)

        x1_2 = self.Maxpool(x1_1)
        x2_2 = self.Maxpool(x2_1)
        x1_2 = self.Conv2(x1_2)
        x2_2 = self.Conv2(x2_2)

        x1_3 = self.Maxpool(x1_2)
        x2_3 = self.Maxpool(x2_2)
        x1_3 = self.Conv3(x1_3)
        x2_3 = self.Conv3(x2_3)
      
        x1_4 = self.Maxpool(x1_3)
        x2_4 = self.Maxpool(x2_3)
        x1_4 = self.Conv4(x1_4)
        x2_4 = self.Conv4(x2_4)
   
        x1_5 = self.Maxpool(x1_4)
        x2_5 = self.Maxpool(x2_4)
        x1_5 = self.Conv5(x1_5)
        x2_5 = self.Conv5(x2_5)

        diff_1 = torch.sub(x1_1, x2_1)
        diff_2 = torch.sub(x1_2, x2_2)
        diff_3 = torch.sub(x1_3, x2_3)
        diff_4 = torch.sub(x1_4, x2_4)
        diff_5 = torch.sub(x1_5, x2_5)


        up_5 = self.up5(x1_5)
        at_4, layerlabel4 = self.att5(diff_4,up_5)
        up_5 = torch.cat((at_4, up_5), dim = 1)
        up_5 = self.Up_conv5(up_5)
 
        up_4 = self.up4(up_5)
        at_3, layerlabel3 = self.att4(diff_3, up_4)
        up_4 = torch.cat((at_3, up_4), dim = 1)
        up_4 = self.Up_conv4(up_4)
    
        up_3 = self.up3(up_4)
        at_2, layerlabel2 = self.att3(diff_2, up_3)
        up_3 = torch.cat((at_2, up_3), dim = 1)
        up_3 = self.Up_conv3(up_3)
     
        up_2 = self.up2(up_3)
        at_1, layerlabel1 = self.att2(diff_1,  up_2)
        up_2 = torch.cat((at_1, up_2), dim = 1)
        up_2 = self.Up_conv2(up_2)
      
        output = self.Conv_1x1(up_2)
    
        blended = self.agr(at_1)
        sig = self.sigmoid(blended)
        mask = torch.where(sig > 0.5, 1, 0)     
   
        return output, mask, [layerlabel4, layerlabel3, layerlabel2]

 
