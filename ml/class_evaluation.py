import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import threading

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from inference import predict_image


# -----------------------------
# CONFIG
# -----------------------------
DATASET_DIR = "raw-img"

model_path = sys.argv[1]
if not model_path:
    print("Error: No model path provided.")
    print("Usage: python class_evaluation.py <model_path>")
    sys.exit(1)


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

# Lock for thread-safe dictionary updates
dict_lock = threading.Lock()


# -----------------------------
# WORKER FUNCTION
# -----------------------------
def process_image(true_class, image_path, model_path):
    """Process a single image and return results."""
    try:
        predicted_class = predict_image(image_path, model_path=model_path)
        return (true_class, predicted_class, None)
    except Exception as e:
        return (true_class, None, str(e))


# -----------------------------
# EVALUATION LOOP (MULTITHREADED)
# -----------------------------
max_workers = 16  # Adjust based on your system (4-8 recommended for GPU)

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    
    for true_class in classes:
        class_folder = os.path.join(DATASET_DIR, true_class)
        print(f"Evaluating class: {true_class}")
        
        # Submit all images for this class
        futures = []
        filenames = os.listdir(class_folder)
        
        for filename in filenames:
            image_path = os.path.join(class_folder, filename)
            future = executor.submit(process_image, true_class, image_path, model_path)
            futures.append(future)
        
        # Collect results as they complete
        for future in as_completed(futures):
            true_class_result, predicted_class, error = future.result()
            
            if error:
                print(f"Could not process image: {error}")
                continue
            
            with dict_lock:
                total_per_class[true_class_result] += 1
                prediction_distribution[true_class_result][predicted_class] += 1
                
                if predicted_class == true_class_result:
                    correct_per_class[true_class_result] += 1


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

    # Save plot to file instead of showing
    output_dir = "evaluation_plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{class_name}_distribution.png"), dpi=100, bbox_inches='tight')
    plt.close()