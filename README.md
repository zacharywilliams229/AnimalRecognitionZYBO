# AnimalRecognitionZYBO
Creates a neural network for predicting animals and runs on a Zybo board

Dataset: https://www.kaggle.com/datasets/alessiocorrado99/animals10/data

This project will utilize both the ARM processor using RTOS and the FPGA. Managing memory and accuracy will be the most important part of this project.

I have chosen this project to gain experience in embedded systems and in fine-tuning neural networks.

# Step 1: CNN creation
# Model Training
## [Model Definition](ml/model.py)
The model must be small so that it can run on the ZYBO without any issues. It includes three layers and filters the image from 128x128 to 16x16. I added BatchNorm2d to improve accuracy without large increases to the model size. I have the dropout set to 0.3 to balance overfitting and noise.
## [Dataset Processing](ml/dataset.py)
90% of the data is used for training, while 10% is reserved for testing. The data is split for each class so that the correct proportion of classes is used for training. There is also random rotation and flipping, which reduce overfitting drastically.
## [Model Training](ml/train.py)
The model training takes place on an NVIDIA GPU, which allows for lower learning rates and more epochs. I started the learning rate at 0.0005, but I also included a scheduler (ReduceLROnPlateau) that cuts the learning rate in half if the training stops. I used 100 epochs, but the two models I chose were from epochs 82 and 97, respectively.

# Models
## Testing Accuracy Model
Accuracy During Training:
- Training accuracy: 84%
- Testing accuracy: 80%

Accuracy during validation (entire dataset):
- butterfly: 94.03% (1986/2112)
- cat: 82.91% (1383/1668)
- chicken: 93.03% (2882/3098)
- cow: 84.41% (1575/1866)
- dog: 92.10% (4479/4863)
- elephant: 88.93% (1286/1446)
- horse: 88.98% (2334/2623)
- sheep: 86.76% (1579/1820)
- spider: 96.20% (4638/4821)
- squirrel: 88.72% (1652/1862)
- overall: 90.89% (23794/26179)

[Plots for each class](/evaluation_plots_val_acc_model)

## Overall Accuracy Model
Accuracy During Training:
- Training accuracy: 85%
- Testing accuracy: 79%

Accuracy during validation (entire dataset):
- butterfly: 93.42% (1973/2112)
- cat: 82.01% (1368/1668)
- chicken: 93.48% (2896/3098)
- cow: 84.62% (1579/1866)
- dog: 92.04% (4476/4863)
- elephant: 89.42% (1293/1446)
- horse: 90.24% (2367/2623)
- sheep: 86.32% (1571/1820)
- spider: 96.41% (4648/4821)
- squirrel: 89.37% (1664/1862)
- overall: 91.05% (23835/26179)

[Plots for each class](/evaluation_plots_overall_acc_model)

## Conclusions
I consider both of these models a success. With ten classes, a model that didn't learn anything would achieve about a 10% accuracy. The baseline presented for homemade CNNs on the dataset is 80% accuracy. One of these models achieved that, and the other was within a percentage. The models are also about 8kB, which should be small enough to be run on the ZYBO. I will use the Testing accuracy model for the rest of my project.

# Step 2: Export Models
In Progress :)
