import torch
from torch import nn, optim
from tqdm import tqdm

# Code for training the model

def train_model(model, train_loader, device, epochs=8):
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training settings
    train_losses = []

    for epoch in range(epochs):
        model.train()  # Training mode
        running_loss = 0

        # Show progress
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit='batch'):
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate average loss for this epoch
        avg_loss = running_loss / len(train_loader)

        # Store the average loss
        train_losses.append(avg_loss)  

        print(f"Epoch {epoch+1}/{epochs} - Training loss: {avg_loss:.4f}")

    return train_losses
