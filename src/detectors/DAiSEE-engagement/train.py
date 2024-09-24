import torch
import torch.optim as optim
import torch.nn as nn
from model import SimpleCNN


def train_model(train_loader, val_loader, num_epochs=5, learning_rate=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)

    # Use BCEWithLogitsLoss for multi-label classification
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)  # Learning rate scheduler

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['video'], data['labels']  # Use all labels now

            # Move inputs and labels to the GPU/CPU
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        # Step the learning rate scheduler
        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                images, labels = data['video'], data['labels']
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                # Calculate validation loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate binary predictions
                predicted = torch.sigmoid(outputs).round()

                # Calculate accuracy (correct multi-label predictions)
                correct += (predicted == labels).sum().item()
                total += labels.numel()  # Total number of labels

        # Calculate average loss and accuracy
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')

    print('Finished Training')
    return model
