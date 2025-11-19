# Collaborative CNN Cross-Dataset Evaluation Report  
## Team 12– Tejas(2025CSZ0008) & Divya(2025AIZ0019)
# 1. Introduction

This project evaluates the **cross-dataset generalization** of two independently trained CNN models.  
Each team member trained a model on their **own dataset** and then tested the model from the other teammate.
 
The objectives of the project are:

1. To analyze how CNN models trained on different datasets perform when applied to unseen data from another user.  
2. To compare the generalization capability of **Model V1 (User-1)** and **Model V2 (User-2)**.  
3. To learn and demonstrate proper GitHub collaboration workflow including:
   - Forking & cloning  
   - Branching  
   - Pull Requests  
   - Issues  
   - Merge workflow  

Both models are implemented using **ResNet-18 (Transfer Learning)** but trained on different datasets, with different augmentations and slightly different fine-tuning approaches.

# 2. Dataset Overview

## **Dataset 1**
https://www.kaggle.com/datasets/tongpython/cat-and-dog

## **Dataset 2**
https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition

### **Key Observation**
Both datasets have the **same number of classes**, but:
- class distribution  
- background variations  
- image quality  
- camera sources  
- illumination  


# 3. Model Architectures
Model V1 
A transfer-learning ResNet-18 model:
self.base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
self.base.fc = nn.Linear(in_features, num_classes=2)

Characteristics:
Pretrained on ImageNet
Fully connected layer replaced
Mild augmentations
Optimizer: Adam
Loss: CrossEntropyLoss

Model V2 
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes=2)

Characteristics:
Similar backbone (ResNet-18)
Trained on a different dataset
Slightly different training dynamics
Fine-tuning strategy varies

# 4. Experimental Results

4.1 Self-Evaluation Metrics
Model V1 on dataset_1
Metric	    Score
Accuracy	  0.9777
Precision	  0.9780
Recall	    0.9777
F1-Score	  0.9777


4.2 Cross-Testing Results
Model V2 tested  dataset_2
Metric	    Score
Accuracy	  0.9810
Precision  	0.9813
Recall	    0.9810
F1-Score	  0.9810

→ Performs better than Model-1 on dataset 1
→ Indicates improved generalization stability


# 5. Observations & Analysis

1. Dataset Influence
Model V2 appears to have been trained on a dataset with more varied samples or stronger augmentations.
This might explain why V2 generalizes better.

2. Transfer Learning Efficiency
Both models used ResNet-18:

Model V1 → stable baseline

Model V2 → slightly more robust, likely due to:
Better augmentation
More diverse dataset
Longer/optimized training

3. Cross-Domain Performance
Model V2 maintains almost identical performance on dataset1 as its self-test score.

4. Conclusion from results
Model V2 has better cross-dataset generalization, suggesting:
It learned more robust features
Or its dataset had better variation
Or training hyperparameters were more optimal

# 6. Conclusion
This collaborative project demonstrates how:

Two similar architectures can behave differently on unseen data

Dataset diversity and augmentation significantly impact generalization

Cross-testing is crucial for evaluating robustness

GitHub workflows (PR, issues, branching) simulate real collaborative ML development.







