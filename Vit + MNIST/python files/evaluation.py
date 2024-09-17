import torch

# Code for evaluating the model's performance

def evaluate_model(model, test_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()  # Evaluation mode
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")
    return accuracy
