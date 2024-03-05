# Brain Tumor Image Classification

This project involves the classification of brain tumor images using Convolutional Neural Networks (CNNs). The dataset consists of images of brain tumors along with various image statistics such as mean, variance, standard deviation, entropy, skewness, kurtosis, contrast, energy, ASM (Angular Second Moment), homogeneity, dissimilarity, correlation, and coarseness. The objective is to build a CNN model that can accurately classify these images into tumor or non-tumor classes based on the image statistics.

## Introduction

Brain tumor classification is a crucial task in medical imaging analysis, aiding in diagnosis and treatment planning. Deep learning techniques, particularly CNNs, have shown promising results in image classification tasks. This project utilizes CNNs to classify brain tumor images based on image statistics.

## Dataset

The dataset used for this project consists of brain tumor images along with image statistics. Each image is labeled with a class indicating whether it contains a tumor (class 1) or not (class 0). The dataset is provided in CSV format and includes image statistics such as mean, variance, standard deviation, entropy, etc.


![image](https://github.com/PiyushChaudhari-007/Brain_TumorDetection/assets/147206358/51db070b-efa9-45da-87ce-21f5a7bb4570)

## Analysis

The analysis involves the following steps:

1. **Data Loading and Exploration**: Loading the dataset into a Pandas DataFrame and exploring its structure and contents.
   
2. **Data Preprocessing**: Preprocessing the image data, including loading the images, resizing them, and normalizing pixel values.

3. **Data Augmentation**: Performing data augmentation to enhance the training dataset and prevent overfitting.

4. **Model Building**: Building a CNN model using Keras with TensorFlow backend. The model architecture consists of convolutional layers followed by max-pooling layers and dense layers.

5. **Model Training**: Training the CNN model on the augmented training dataset.

6. **Model Evaluation**: Evaluating the trained model on the test dataset and analyzing performance metrics such as accuracy, loss, and confusion matrix.

## Code Implementation

The code for this project is implemented in Python using libraries such as Pandas, NumPy, Matplotlib, TensorFlow, and Keras. The Jupyter notebook or Python script provides step-by-step instructions for loading data, preprocessing, model building, training, and evaluation.
## Visualizing The Process
Single Convolution
![gif](https://github.com/PiyushChaudhari-007/Brain_TumorDetection/assets/147206358/eb19d82b-34e5-4cff-ba06-06d2951763a0)


## Conclusion

The trained CNN model demonstrates promising results in classifying brain tumor images based on image statistics. Further optimization and fine-tuning of the model could potentially improve its performance and make it more suitable for real-world applications in medical imaging analysis. 

![image](https://github.com/PiyushChaudhari-007/Brain_TumorDetection/assets/147206358/863041e1-cc99-4229-a65a-4a1babbcf2c3)

(![image](https://github.com/PiyushChaudhari-007/Brain_TumorDetection/assets/147206358/d35666c5-50b1-4b1f-856c-d64fc995a8c9)

This confusion matrix illustrates the classification performance of the model, showing the number of true positives, true negatives, false positives, and false negatives. The heatmap visualization provides a clear representation of the model's predictions compared to the ground truth labels.
