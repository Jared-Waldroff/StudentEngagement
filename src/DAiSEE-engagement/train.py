# train.py
import torch
import torch.optim as optim
import torch.nn as nn
from model import SimpleCNN

def train_model(train_loader, val_loader, num_epochs=5, learning_rate=0.001):
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['video'], data['engagement']  # Adjust based on your task

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100}')
                running_loss = 0.0

    print('Finished Training')

    # Validation
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data['video'], data['engagement']  # Adjust based on your task
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the validation set: {100 * correct / total}%')

    return model
