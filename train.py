import copy
#import matplotlib.pyplot as plt 
import numpy as np
import os
#import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from datetime import datetime
from PIL import Image,ImageDraw, ImageFont
#from sklearn.metrics import confusion_matrix
#from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader,Dataset
import cv2 

from model import load_model, classes


class ImageDataset(Dataset):
    def __init__(self, folder_path, classes, transform=None):
        names = os.listdir(folder_path) 
        self.filepaths = []
        self.y = []
        for c in classes.keys():
            folder_class = os.path.join(folder_path,c)
            names = os.listdir(folder_class)
            f = [os.path.join(folder_path,c,name) for name in names]
            self.filepaths.extend(f)
            self.y.extend([classes[c] for i in range(0,len(f))])
        self.transform = transform
    def __getitem__(self, index):
        filepath = self.filepaths[index]
        y = self.y[index]
        x = Image.open(filepath).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x,y
    def __len__(self):
        return len(self.filepaths)


# validation metric classification
def metrics_func_classification(target, output):
    # Compute number of correct prediction
    pred = output.argmax(dim=-1, keepdim=True)
    corrects = pred.eq(target.reshape(pred.shape)).sum().item()
    return -corrects # minus for coeherence with best result is the most negative one


# training: loss calculation and backward step
def loss_batch(loss_func, metric_func, xb, yb, yb_h, opt=None):
    # obtain loss
    loss = loss_func(yb_h, yb)
    # obtain performance metric
    with torch.no_grad():
        metric_b = metric_func(yb, yb_h)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), metric_b


# one epoch training
def loss_epoch(model, loss_func, metric_func, dataset_dl, sanity_check, opt, device):
    loss = 0.0
    metric = 0.0
    len_data = float(len(dataset_dl.dataset))
    # get batch data
    for xb, yb in dataset_dl:    
        # send to cuda the data (batch size)
        xb = xb.to(device)
        yb = yb.to(device)
        # obtain model output 
        yb_h = model.forward(xb)
        # loss and metric Calculation
        loss_b, metric_b = loss_batch(loss_func, metric_func, xb, yb, yb_h, opt)
        # update loss
        loss += loss_b
        # update metric
        if metric_b is not None:
            metric+=metric_b 
        if sanity_check is True:
            break
    # average loss
    loss /= len_data
    # average metric
    metric /= len_data
    return loss, metric


# get learning rate from optimizer
def get_lr(opt):
    # opt.param_groups[0]['lr']
    for param_group in opt.param_groups:
        return param_group["lr"]


# trainig - test loop
def train_test(params):
    # --> extract params
    model = params["model"]
    loss_func = params["loss_func"]
    metric_func = params["metric_func"]
    num_epochs = params["num_epochs"]
    opt = params["optimizer"]
    lr_scheduler = params["lr_scheduler"]
    train_dl = params["train_dl"]
    test_dl = params["test_dl"]
    device = params["device"]
    continue_training = params["continue_training"]
    sanity_check = params["sanity_check"]
    path2weigths = params["path2weigths"]
    # --> send model to device and print device
    model = model.to(device)
    print("--> training device %s" % (device))
    # --> if continue_training=True load path2weigths
    if continue_training==True and os.path.isfile(path2weigths):
        print("--> continue training  from last best weights")
        weights = torch.load(path2weigths)
        model.load_state_dict(weights)
    # --> history of loss values in each epoch
    loss_history={"train": [],"test":[]}
    # --> history of metric values in each epoch
    metric_history={"train": [],"test":[]}
    # --> a deep copy of weights for the best performing model
    best_model_weights = copy.deepcopy(model.state_dict())
    # --> initialiaze best loss to large value
    best_loss = float("inf")
    # --> main loop
    for epoch in range(num_epochs):
        # --> get learning rate
        lr = get_lr(opt)
        print("----\nEpoch %s/%s, lr=%.6f" % (epoch+1,num_epochs,lr))
        # --> train model on training dataset
        # we tell to the model to enter in train state. it is important because
        # there are somelayers like dropout, batchnorm that behaves 
        # differently between train and test
        model.train()
        train_loss,train_metric = loss_epoch(model, loss_func, metric_func, train_dl, sanity_check, opt, device)
        # --> collect loss and metric for training dataset
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        # --> tell the model to be in test (validation) mode
        model.eval()
        with torch.no_grad():
            test_loss, test_metric = loss_epoch(model, loss_func, metric_func, test_dl, sanity_check, opt=None, device=device)
        # --> collect loss and metric for test dataset
        loss_history["test"].append(test_loss)
        metric_history["test"].append(test_metric)
        # --> store best model
        if test_loss < best_loss:
            print("--> model improved! --> saved to %s" %(path2weigths))
            best_loss = test_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            # --> store weights into local file
            torch.save(model.state_dict(), path2weigths)
        # --> learning rate scheduler
        lr_scheduler.step()
        print("--> train_loss: %.6f, test_loss: %.6f, train_metric: %.3f, test_metric: %.3f" % (train_loss,test_loss,train_metric,test_metric))
    # --> load best weights
    model.load_state_dict(best_model_weights)
    return model, loss_history, metric_history
        


if __name__ == "__main__":

    # Generate Train test folder
    folder_train_path = os.path.join("data","images","train")
    folder_test_path = os.path.join("data", "images","val")


    # -----------  Read Data
    class_names =  list(classes.keys())
    num_classes = len(class_names)
    size = 224 
    # -----------  

    # ----------- Count images per classes
    # Count number of images
    #train = {}
    #test = {}
    #for c in class_names:
    #    train[c] = len(os.listdir(os.path.join(folder_train_path,c)))
    #    test[c] = len(os.listdir(os.path.join(folder_test_path,c)))
    #print("Number of images per class -> Train:", train, "Test:", test)
    # -----------  

    # ---------- Data loader 
    size = 224 # AlexNet-SqueezeNet-VGG16-Resent18 (224,224)
    train_transform = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.RandomApply(torch.nn.ModuleList([
                transforms.RandomAffine(degrees=(1, 10), translate=(0.01, 0.03), scale=(1.1, 1.3))
                ]),
            p=0.4,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    test_transform = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = ImageDataset(folder_train_path, classes, train_transform)
    test_ds = ImageDataset(folder_test_path, classes, test_transform)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=50, shuffle=True)
    test_dl = torch.utils.data.DataLoader(train_ds, batch_size=50, shuffle=True)
    #train_dl = torch.utils.data.DataLoader(train_ds, batch_size=50, shuffle=True)
    #for x,y in train_dl: 
    #    batch_grid = torchvision.utils.make_grid(x, nrow=5, padding=5)
    #    im = transforms.ToPILImage()(batch_grid)
    #    cv2.imshow("Image", im)
    #    cv2.waitKey(0)
    #    break
    # -----------  

    # -----------  Load Network
    net = load_model(pretrained=True, num_classes=num_classes)
    # -----------

    # -----------  Train
    # Setup GPU Device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    opt = optim.Adam(net.parameters(), lr=0.0001)
    train_dl = DataLoader(train_ds, batch_size=20, shuffle=True, num_workers=1)
    test_dl = DataLoader(test_ds, batch_size=20, shuffle=True, num_workers=1)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.999)  #  lr = lr * gamma ** last_epoch
    params = {
        "model":                 net,
        "loss_func":             nn.NLLLoss(reduction="sum"),
        "metric_func":           metrics_func_classification,
        "num_epochs":            15,
        "optimizer":             opt,
        "lr_scheduler":          lr_scheduler,
        "train_dl":              train_dl,
        "test_dl":               test_dl,
        "device":                device,
        "continue_training" :    False, # continue training from last save weights
        "sanity_check":          False, # if true we only do one batch per epoch
        "path2weigths":          "models/net.pt"
    } 
    model, loss_history, metric_history = train_test(params)