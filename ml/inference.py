import torch
import sys
from torchvision import transforms
from PIL import Image

from model import AnimalCNN


# Class names in EXACT order used during training
classes = [
    "butterfly",
    "cat",
    "chicken",
    "cow",
    "dog",
    "elephant",
    "horse",
    "sheep",
    "spider",
    "squirrel"
]


def predict_image(image_path, model_path="ml/models/animalcnn.pth"):

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = AnimalCNN(num_classes=len(classes))

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Move to device
    model.to(device)

    # Evaluation mode
    model.eval()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        )
    ])

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Transform image
    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)

    # Move to device
    image = image.to(device)

    # Disable gradients
    with torch.no_grad():

        outputs = model(image)

        # Get predicted class index
        _, predicted = torch.max(outputs, 1)

        predicted_class = classes[predicted.item()]

    print(f"Prediction: {predicted_class}")


if __name__ == "__main__":

    image_path = sys.argv[1]
    if not image_path:
        print("Usage: python inference.py <image_path> [model_path]")
        sys.exit(1)
    model_path = sys.argv[2] if len(sys.argv) > 2 else "ml/models/animalcnn.pth"

    predict_image(image_path, model_path)