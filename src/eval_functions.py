import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, Dict

def evaluate_model(model: torch.nn.Module, 
                test_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate the model's performance on the test set. Calculate and return accuracy
    """
    device = next(model.parameters()).device
    model.eval()
    
    correct = 0
    total = 0
    class_correct = {i: 0 for i in range(10)}
    class_total = {i: 0 for i in range(10)}
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    class_correct[label.item()] += 1
                class_total[label.item()] += 1
    
    # Calculate overall accuracy
    accuracy = 100 * correct / total
    
    # Calculate per-class accuracy
    class_accuracies = {
        class_name: (100 * class_correct[i] / class_total[i])
        for i, class_name in enumerate([
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ])
    }
    
    # Store metrics
    metrics = {
        'overall_accuracy': accuracy,
        'per_class_accuracy': class_accuracies
    }
    
    return accuracy, metrics

def save_evaluation_results(metrics: Dict, config: 'TrainingConfig', save_path: str):
    """Save evaluation results to a file."""
    with open(save_path, 'w') as f:
        f.write("Model Evaluation Results\n")
        f.write("========================\n\n")
        f.write("Configuration:\n")
        f.write(f"- Batch Size: {config.batch_size}\n")
        f.write(f"- Epochs: {config.num_epochs}\n")
        f.write(f"- Learning Rate: {config.learning_rate}\n\n")
        
        f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%\n\n")
        
        f.write("Per-Class Accuracy:\n")
        for class_name, accuracy in metrics['per_class_accuracy'].items():
            f.write(f"- {class_name}: {accuracy:.2f}%\n")

def plot_class_accuracies(metrics: Dict, save_path: str):
    """Create and save a bar plot of per-class accuracies."""
    class_accuracies = metrics['per_class_accuracy']
    
    plt.figure(figsize=(12, 6))
    plt.bar(class_accuracies.keys(), class_accuracies.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('Accuracy by Fashion MNIST Class')
    plt.ylabel('Accuracy (%)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()