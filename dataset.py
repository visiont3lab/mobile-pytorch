import copy
# import matplotlib.pyplot as plt
import numpy as np
import os
# import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader, Dataset
import cv2

from model import load_model, classes


class ImageDataset(Dataset):
    def __init__(self, folder_path, classes, transform=None):
        names = os.listdir(folder_path)
        self.filepaths = []
        self.y = []
        for c in classes.keys():
            folder_class = os.path.join(folder_path, c)
            names = os.listdir(folder_class)
            f = [os.path.join(folder_path, c, name) for name in names]
            self.filepaths.extend(f)
            self.y.extend([classes[c] for i in range(0, len(f))])
        self.transform = transform

    def __getitem__(self, index):
        filepath = self.filepaths[index]
        y = self.y[index]
        x = Image.open(filepath).convert("RGB")
        #x = x.rotate(-90, expand=True)  # TODO issue with image rotation
        #x = x.crop((0,500,3000,2405))
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.filepaths)

def get_data_loaders():
    # Generate Train test folder
    folder_train_path = os.path.join("data", "dataset-crop", "train")
    folder_test_path = os.path.join("data", "dataset-crop", "test")

    # -----------  Read Data
    class_names = list(classes.keys())
    num_classes = len(class_names)
    size = 224
    # -----------

    # ----------- Count images per classes
    # Count number of images
    test = {}
    train = {}
    for c in class_names:
        train[c] = len(os.listdir(os.path.join(folder_train_path, c)))
        test[c] = len(os.listdir(os.path.join(folder_test_path, c)))
    print("Number of images per class -> Train:", train, "Test:", test)
    # -----------

    # ---------- Data loader
    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomApply(torch.nn.ModuleList([
            transforms.RandomAffine(degrees=(1, 5), translate=(0.01, 0.1), scale=(1.1, 1.3)),
            transforms.ColorJitter(brightness=.1, contrast=.25),
        ]),
            p=0.4,
        ),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = ImageDataset(folder_train_path, classes, train_transform)
    test_ds = ImageDataset(folder_test_path, classes, test_transform)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=4, shuffle=True)

    return train_dl, test_dl


if __name__ == "__main__":
    train_dl, test_dl = get_data_loaders()

    for x, y in train_dl:
        batch_grid = torchvision.utils.make_grid(x, nrow=8, padding=5)
        im = np.array(transforms.ToPILImage()(batch_grid))
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", im)
        cv2.waitKey(0)
        break