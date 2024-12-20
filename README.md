# Fashion MNIST CNN Classification Project

A modular PyTorch implementation for training and evaluating CNNs on the Fashion MNIST dataset.

## Project Structure

```
fashion-mnist-project/
├── src/                 # Source code directory
│   ├── __init__.py     # Makes src a Python package
│   ├── config.py       # Hyperparameter configuration
│   ├── model.py        # CNN model architecture
│   ├── train.py        # Training logic
│   └── evaluate.py     # Evaluation and metrics
├── data/               # Dataset storage
├── models/             # Saved model checkpoints
├── results/            # Evaluation results & plots
└── main.py            # Entry point script
```

## Key Benefits

### 1. Modular Design
- **Independent Components**: Each module has a single responsibility
- **Easy Maintenance**: Simple to update or modify individual components
- **Reusability**: Components can be used in other projects
- **Testing**: Modular structure facilitates unit testing

### 2. Configuration Management
- Centralized hyperparameter control in `config.py`
- Easy experimentation with different parameters
- Clear tracking of experimental configurations
- Simple parameter adjustment without code changes

### 3. Model Architecture
- Clean separation of model architecture in `model.py`
- Easy to modify or swap network architectures
- Independent model testing and validation
- Clear structure for implementing new models

### 4. Training Pipeline
- Organized training process in `train.py`
- Flexible data loading system
- Easy to modify training procedures
- Support for different optimization strategies

### 5. Evaluation System
- Comprehensive evaluation metrics in `evaluate.py`
- Automated result saving and visualization
- Easy to add new evaluation metrics
- Clear reporting of model performance

### 6. Data Organization
- **data/**: Organized dataset storage
- **models/**: Systematic model checkpoint saving
- **results/**: Clear storage of evaluation results
- Easy to manage with version control

## Getting Started

1. **Installation**
```bash
pip install -r requirements.txt
```

2. **Training**
```bash
python main.py
```

3. **Configuration**
Edit hyperparameters in `src/config.py`:
```python
config = TrainingConfig(
    batch_size=64,
    num_epochs=3,
    learning_rate=0.01
)
```

## Features

- Configurable CNN architecture
- Flexible hyperparameter management
- Automated model evaluation
- Performance visualization
- Result logging and storage

## Requirements

- Python 3.12+
- PyTorch 2.0+
- torchvision
- matplotlib
