# Training Model for Mobile using Pytorch Mobile

```
cd mobile-pytorch
python3 -m venv env
# or virtualenv --python=python3.8 env
pip install torch torchvision opencv-python

# Get the dataset

# Train the model
python train.py

# Get mobile optmized model
python optmize.py

# Test optimize model vs torch model
python test.py
```