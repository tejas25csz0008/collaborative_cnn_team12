Collaborative CNN Project

**Overview**
This project is a 2-user collaboration to train and evaluate CNN models on different cat–dog datasets.
User 1 trains Model v1, User 2 trains Model v2, and both test each other’s models.


**1. Set up repository**
Folders:
models/
notebooks/
results/
utils/

**2. Trained Model v1**
Run: python notebooks/train_v1.py
Saved model: models/model_v1.pth

**3. Tested Model v1 (User1 dataset)**
Run: python notebooks/test_v1.py
Saved metrics: results/test_v1_metrics.json

**4. Git Workflow**
Created branch: git checkout -b dev_user1
