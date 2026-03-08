# Plant-Disease-Recognition-Model using Transfer Learning

This project focuses on building an image classification model to recognize various plant diseases using deep learning, specifically leveraging transfer learning with a pre-trained EfficientNetB4 model.

Project Objective
The main goal of this project is to develop a robust model capable of accurately identifying different plant diseases from leaf images, which can aid in early detection and treatment in agriculture.

Dataset
The dataset used for this project is sourced from Mendeley Data, titled 'Plant leave diseases dataset with augmentation'. It contains a comprehensive collection of augmented images across 39 different plant disease classes and healthy plant leaves. The dataset was downloaded and unzipped within the Colab environment.

Methodology
The project employs a transfer learning approach, utilizing the powerful EfficientNetB4 architecture pre-trained on the ImageNet dataset. The key steps involved are:

Data Preparation:

The dataset was split into training (80%), validation (10%), and test (10%) sets using the split-folders library.
Images were preprocessed by resizing to 160x160 pixels and organized into tf.data.Dataset objects for efficient loading and augmentation.
AUTOTUNE was used for prefetching data to optimize input pipeline performance.
Model Architecture:

An EfficientNetB4 model, pre-trained on ImageNet, was used as the base convolutional layer. Its top (classification) layers were removed (include_top=False).
A GlobalAveragePooling2D layer was added to reduce the spatial dimensions of the features.
A Dropout layer (with a rate of 0.2) was included for regularization.
A Dense output layer with sigmoid activation was added, matching the number of disease classes (39).
Training Strategy (Transfer Learning):

Feature Extraction (Initial Training): The pre-trained EfficientNetB4 base model was initially frozen (base_model.trainable = False), and only the newly added classification layers were trained for 10 epochs. This step allowed the model to learn to classify the specific disease patterns based on the extracted features from the frozen base.
Fine-tuning: After the initial training, a portion of the base model's layers (from layer 100 onwards) was unfrozen (base_model.trainable = True). The entire model was then trained for an additional 10 epochs with a lower learning rate. This fine-tuning step allowed the model to adapt the pre-trained features more specifically to the plant disease dataset.
Optimization:

The model was compiled with the Adam optimizer, SparseCategoricalCrossentropy loss function, and SparseCategoricalAccuracy as a metric.
Results
After 20 epochs of training (10 feature extraction + 10 fine-tuning), the model achieved the following performance:

Test Accuracy: 95.48%
The training and validation accuracy and loss curves demonstrate effective learning and convergence.
