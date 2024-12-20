import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from .model import FashionCNN
from .config import TrainingConfig

def get_data_loaders(config):
    """Create data loaders with current batch size"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    trainset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=True,
        download=True,
        transform=transform
    )
    
    testset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        testset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader

def train_model(config=None):
    """Train the model with given configuration"""
    if config is None:
        config = TrainingConfig()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data loaders with current batch size
    train_loader, test_loader = get_data_loaders(config)
    
    # Initialize model, criterion, and optimizer
    model = FashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{config.num_epochs}], Loss: {avg_loss:.4f}')
    
    return model, test_loader