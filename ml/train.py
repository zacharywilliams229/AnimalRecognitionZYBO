import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

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
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3
    )
    # 6. Training loop
    epochs = 30

    for epoch in range(epochs):

        model.train()  # training mode

        running_loss = 0.0
        correct = 0
        total = 0
        train_pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}", leave=False)

        for images, labels in train_pbar:

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

            # show batch metrics
            batch_acc = 100 * (predicted == labels).sum().item() / labels.size(0)
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", batch_acc=f"{batch_acc:.2f}%")

        train_acc = 100 * correct / total

        # 7. Validation step
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Val {epoch+1}/{epochs}", leave=False)

            for images, labels in val_pbar:

                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                batch_acc = 100 * (predicted == labels).sum().item() / labels.size(0)
                val_pbar.set_postfix(batch_acc=f"{batch_acc:.2f}%")

        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Loss: {running_loss:.4f}")
        print(f"Train Acc: {train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%")
        print("-" * 30)

        # Step the scheduler
        scheduler.step(val_acc)


    # 8. Save model
    torch.save(model.state_dict(), "models/animalcnn.pth")
    print("Model saved.")

if __name__ == "__main__":
    train()