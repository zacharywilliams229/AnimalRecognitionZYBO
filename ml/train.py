import torch
import torch.nn as nn
import torch.optim as optim

from model import AnimalCNN
from dataset import get_dataloaders


def train():

    # 1. Device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load data
    train_loader, val_loader = get_dataloaders()

    # 3. Load model
    model = AnimalCNN(num_classes=10)  # change based on your dataset
    model.to(device)

    # 4. Loss function
    criterion = nn.CrossEntropyLoss()

    # 5. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 6. Training loop
    epochs = 10

    for epoch in range(epochs):

        model.train()  # training mode

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            # move data to device
            images, labels = images.to(device), labels.to(device)

            # zero gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(images)

            # compute loss
            loss = criterion(outputs, labels)

            # backward pass
            loss.backward()

            # update weights
            optimizer.step()

            # stats
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        # 7. Validation step
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Loss: {running_loss:.4f}")
        print(f"Train Acc: {train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%")
        print("-" * 30)


    # 8. Save model
    torch.save(model.state_dict(), "models/animalcnn.pth")
    print("Model saved.")

if __name__ == "__main__":
    train()