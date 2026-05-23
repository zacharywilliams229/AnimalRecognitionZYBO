import os
from collections import defaultdict

import matplotlib.pyplot as plt

from inference import predict_image


# -----------------------------
# CONFIG
# -----------------------------
DATASET_DIR = "raw-img"


# -----------------------------
# LOAD CLASSES
# -----------------------------
classes = sorted([
    folder for folder in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, folder))
])


# -----------------------------
# TRACKING
# -----------------------------
correct_per_class = defaultdict(int)
total_per_class = defaultdict(int)

prediction_distribution = {
    class_name: defaultdict(int)
    for class_name in classes
}


# -----------------------------
# EVALUATION LOOP
# -----------------------------
for true_class in classes:

    class_folder = os.path.join(DATASET_DIR, true_class)

    print(f"Evaluating class: {true_class}")

    for filename in os.listdir(class_folder):

        image_path = os.path.join(class_folder, filename)

        try:
            predicted_class = predict_image(image_path, model_path="ml/models/animalcnn_73.pth")
        except Exception as e:
            print(f"Could not process {image_path}: {e}")
            continue

        total_per_class[true_class] += 1

        prediction_distribution[true_class][predicted_class] += 1

        if predicted_class == true_class:
            correct_per_class[true_class] += 1


# -----------------------------
# PRINT ACCURACIES
# -----------------------------
print("PER-CLASS ACCURACY")

for class_name in classes:

    total = total_per_class[class_name]
    correct = correct_per_class[class_name]

    if total == 0:
        accuracy = 0
    else:
        accuracy = 100 * correct / total

    print(f"{class_name}: {accuracy:.2f}% ({correct}/{total})")


# -----------------------------
# PIE CHARTS
# -----------------------------
for class_name in classes:

    distribution = prediction_distribution[class_name]

    labels = list(distribution.keys())
    sizes = list(distribution.values())

    plt.figure(figsize=(7, 7))

    plt.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%'
    )

    plt.title(f"Predictions for TRUE class: {class_name}")

    plt.show()