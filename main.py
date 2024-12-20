from src.train import train_model
from src.config import TrainingConfig
from src.eval_functions import evaluate_model, save_evaluation_results, plot_class_accuracies
import torch

def main():
    # Define different configurations to try
    configs = [
        TrainingConfig(batch_size=32, num_epochs=5),
        TrainingConfig(batch_size=64, num_epochs=3),
        TrainingConfig(batch_size=128, num_epochs=2)
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\nTraining Configuration {i}:")
        print(f"Batch Size: {config.batch_size}")
        print(f"Epochs: {config.num_epochs}")
        
        # Train model with current configuration
        model, test_loader = train_model(config)
        
        # Save model with configuration-specific name
        model_path = f'models/fashion_mnist_model_batch{config.batch_size}_epoch{config.num_epochs}.pth'
        torch.save(model.state_dict(), model_path)
        
        # Evaluate model
        accuracy, metrics = evaluate_model(model, test_loader)
        
        # Save evaluation results
        save_evaluation_results(
            metrics,
            config,
            f'results/evaluation_batch{config.batch_size}_epoch{config.num_epochs}.txt'
        )
        
        # Create visualization
        plot_class_accuracies(
            metrics,
            f'results/class_accuracies_batch{config.batch_size}_epoch{config.num_epochs}.png'
        )
        
        print(f"Training completed for configuration {i}")
        print(f"Model saved to: {model_path}")
        print(f"Overall accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main()