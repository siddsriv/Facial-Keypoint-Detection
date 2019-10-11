import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv2bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv3bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 5)
        self.conv4bn = nn.BatchNorm2d(256)
        #begin fc layers
        self.fc1 = nn.Linear(256*10*10, 2048) #256 feature maps after final convolution *10*10(height*width of each FM)
        self.fc1bn = nn.BatchNorm1d(2048)
        self.drop6 = nn.Dropout(p=0.6) 
        self.fc2 = nn.Linear(2048, 1024)
        self.fc2bn = nn.BatchNorm1d(1024)
        self.drop7 = nn.Dropout(p=0.6)
        self.fc3 = nn.Linear(1024, 136)




    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2bn(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.conv3bn(self.conv3(x))))
        x = self.pool(F.leaky_relu(self.conv4bn(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1bn(self.fc1(x))) #Try and measure how much better leaky relu is than relu
        x = self.drop6(x)
        x = F.leaky_relu(self.fc2bn(self.fc2(x)))
        x = self.drop7(x)
        x = self.fc3(x)
        return x
