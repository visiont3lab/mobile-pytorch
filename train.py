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
import pandas as pd
from sklearn.metrics import confusion_matrix

from model import load_model, classes
from dataset import get_data_loaders

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
    # --> initialiaze best metric to large value
    best_metric = float("inf")
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
        #if test_metric < best_metric:
        #if train_metric < best_metric:
            print("--> model improved! --> saved to %s" %(path2weigths))
            best_loss = test_loss
            #best_metric = test_metric
            #best_metric = train_metric
            best_model_weights = copy.deepcopy(model.state_dict())
            # --> store weights into local file
            torch.save(model.state_dict(), path2weigths)
        # --> learning rate scheduler
        lr_scheduler.step()
        print("--> train_loss: %.6f, test_loss: %.6f, train_metric: %.3f, test_metric: %.3f" % (train_loss,test_loss,train_metric,test_metric))
    # --> load best weights
    model.load_state_dict(best_model_weights)
    return model, loss_history, metric_history
        

def train():
    # -----------  Get data loaders
    train_dl, test_dl = get_data_loaders()

    # -----------  Load Network
    net = load_model(pretrained=True, num_classes=len(classes))
    
    #num_classes = len(classes)
    #net = load_model(pretrained=True, num_classes=num_classes)
    #path2weights = f"./models/resnet50.pt"
    #weights = torch.load(path2weights)
    #net.load_state_dict(weights)
    # -----------

    # -----------  Train
    # Setup GPU Device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    print("Running on ", device)
    opt = optim.Adam(net.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.999)  # lr = lr * gamma ** last_epoch
    params = {
        "model": net,
        "loss_func": torch.nn.CrossEntropyLoss(reduction="sum"),  # NLLLoss(reduction="sum"),
        "metric_func": metrics_func_classification,
        "num_epochs": 30,
        "optimizer": opt,
        "lr_scheduler": lr_scheduler,
        "train_dl": train_dl,
        "test_dl": test_dl,
        "device": device,
        "continue_training": False,  # continue training from last save weights
        "sanity_check": False,  # if true we only do one batch per epoch
        "path2weigths": "models/resnet50_crop.pt"
    }
    model, loss_history, metric_history = train_test(params)


def test():
    # -----------  Get data loaders
    train_dl, test_dl = get_data_loaders()

    num_classes = len(classes)
    net = load_model(pretrained=True, num_classes=num_classes)
    path2weights = f"./models/resnet50_crop.pt"
    weights = torch.load(path2weights)
    net.load_state_dict(weights)

    # Setup GPU Device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    # Send model to device
    net.to(device)

    # Tell the model layer that we are going to use the model in evaluation  mode!
    net.eval()

    # Predict Classication
    cm = np.zeros((num_classes, num_classes))
    names_pred = ["Pred: " + n for n in classes]
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device)
            out = net.forward(x)
            out = torch.softmax(out, dim=-1)
            y_hat = out.argmax(dim=-1, keepdim=True).cpu().numpy().reshape(-1)
            #print(out.cpu().numpy())
            # Visualize results
            cm += confusion_matrix(y, y_hat)
    print("Confusion Matrix")
    df = pd.DataFrame(cm, columns=names_pred, index=classes)
    print(df)

if __name__ == "__main__":

    train()
    #test()