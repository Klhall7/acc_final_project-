import torch
from torch import nn

# Define the Fashion MNIST Convolutional Neural Network

class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        # Convolutional layers w/parameters (weights and biases)
        
        self.conv_layers = nn.Sequential(
            # First layer (1 input channel, 8 filters, 5x5 kernel)
            nn.Conv2d(1, 8, kernel_size=5),  # Default stride and padding
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second layer
            nn.Conv2d(8, 16, kernel_size=5),  
            # Default stride and padding
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 10)  
            #10 classes in Fashion MNIST
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        # Flatten tensor
        x = x.view(x.size(0), -1)  
        x = self.fc_layers(x)
        return x