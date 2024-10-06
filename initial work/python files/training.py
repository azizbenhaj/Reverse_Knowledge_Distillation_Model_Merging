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

def train_knowledge_distillation(teacher, student, train_loader, epochs=5, learning_rate=0.001, T=2, lambda_param=0.5, device=torch.device("mps")):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.to(device)
    student.to(device)
    teacher.eval()  # evaluation mode
    student.train() # training mode

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit='batch'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model (gradients not saved)
            with torch.no_grad():
                teacher_logits = teacher(images)

            # Forward pass with the student model
            student_logits = student(images)

            soft_teacher = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_student = nn.functional.log_softmax(student_logits / T, dim=-1)

            distillation_fct = nn.KLDivLoss(reduction="batchmean")
            distillation_loss = distillation_fct(soft_student, soft_teacher) * (T ** 2)

            # Calculate the true label loss
            #_, student_predicted = torch.max(student_logits, 1)
            #student_predicted = student_predicted.to(device)
            ce_loss = F.cross_entropy(student_logits, labels)

            # Weighted sum of the two losses
            final_loss = (1. - lambda_param) * ce_loss + lambda_param * distillation_loss

            final_loss.backward()
            optimizer.step()

            running_loss += final_loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")