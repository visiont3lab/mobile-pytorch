import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import timeit
from torchcam.methods import CAM, LayerCAM, SmoothGradCAMpp
from model import load_model, classes
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import  to_pil_image

# https://github.com/frgfm/torch-cam
# https://pytorch.org/tutorials/beginner/introyt/captumyt.html

class Classifier:

    def __init__(self, path2model, device, net_size):
        # Load model (torchscript support)
        # device "cpu", "cuda:0"
        self.net = torch.jit.load(path2model)
        # Setup GPU Device
        self.device = torch.device(device)
        # Send model to device
        self.net.to(self.device)
        # Tell the model layer that we are going to use the model in evaluation  mode!
        self.net.eval()
        # Set net size
        self.net_size = net_size

    def crop_image(self, im_pil):
        im = np.array(im_pil)
        im = im[654:794, 586:1883]  # h,w
        return Image.fromarray(im)

    def preprocess_image(self, im_pil):
        c, h, w = self.net_size
        # im_pil = self.crop_image(im_pil)
        # im_pil_resize = im_pil.resize((h,w))
        im_tensor = transforms.ToTensor()(im_pil).unsqueeze(0)  # scale 0-1
        # im_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(im_tensor)
        im_np = np.array(im_pil)
        return im_tensor, im_pil, im_np

    def predict(self, im_pil):
        with torch.no_grad():
            im_tensor, im_pil, im_np = self.preprocess_image(im_pil)
            im_tensor = im_tensor.to(self.device)
            out = self.net.forward(im_tensor)  # 1xCxHxW
            # use softmax to retreive probabilities
            out = torch.softmax(out, dim=1).reshape(-1)
            val = out.argmax(dim=-1, keepdim=True).item()
            prob = torch.max(out).item()
            return prob, val

    def visualize(self, im_pil, val, prob):
        # draw on images
        # im_pil = self.crop_image(im_pil)
        im_np = np.array(im_pil)
        cv2.putText(im_np, classes[val] + ": %.2f" % (prob), (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 1,
                    cv2.LINE_AA)
        display(Image.fromarray(im_np))

    def save(self, result_folder_path, im_pil, val, prob):
        # draw on images
        # im_pil = self.crop_image(im_pil)
        im_np = np.array(im_pil)
        im_np = cv2.cvtColor(im_np, cv2.COLOR_BGR2RGB)
        cv2.putText(im_np, classes[val] + ": %.2f" % (prob), (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 1,
                    cv2.LINE_AA)
        cv2.imwrite(os.path.join(result_folder_path, f"{classes[val]}_{name}"), im_np)


class ClassifierAct:

    def __init__(self, path2model, device, net_size):
        # Load model (torchscript support)
        # device "cpu", "cuda:0"
        self.net = load_model(pretrained=False, num_classes=len(classes))
        path2weights = path2model
        self.net.load_state_dict(torch.load(path2weights))

        #print(self.net)

        # Setup GPU Device
        self.device = torch.device(device)
        # Send model to device
        self.net.to(self.device)
        # Tell the model layer that we are going to use the model in evaluation  mode!
        self.net.eval()
        # Set net size
        self.net_size = net_size

        #self.cam_extractor = CAM(self.net, 'layer4', 'fc')  # use layer beform avgpoll!
        #self.cam_extractor = CAM(self.net, 'layer4')
        self.cam_extractor = LayerCAM(self.net, 'layer4')
        

    def crop_image(self, im_pil):
        im = np.array(im_pil)
        im = im[654:794, 586:1883]  # h,w
        return Image.fromarray(im)

    def preprocess_image(self, im_pil):
        c, h, w = self.net_size
        # im_pil = self.crop_image(im_pil)
        im_pil = im_pil.resize((h,w))
        im_tensor = transforms.ToTensor()(im_pil).unsqueeze(0)  # scale 0-1
        # im_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(im_tensor)
        im_np = np.array(im_pil)
        return im_tensor, im_pil, im_np

    def predict(self, im_pil):
        #with torch.no_grad():
        im_tensor, im_pil, im_np = self.preprocess_image(im_pil)
        im_tensor = im_tensor.to(self.device)
        out = self.net.forward(im_tensor)  # 1xCxHxW
        return out

    def get_activation_map(self, im_pil, out):
        # Activation map
        im_np = np.array(im_pil)
        activation_map_list = self.cam_extractor(out.squeeze(0).argmax().item(), out)
        activation_map = activation_map_list[0][0]
        if self.device == torch.device("cuda:0"):
            cmap = activation_map.cpu().numpy()
        else:
            cmap = activation_map.numpy()

        map_np = np.array( overlay_mask(im_pil, to_pil_image(cmap, mode='F'), alpha=0.5) ) # RGB
        
        #cmap = cv2.resize(cmap, (im_np.shape[1], im_np.shape[0]))
        #cmap = (cmap - np.min(cmap) ) / (np.max(cmap) - np.min(cmap))
        #cmap = np.uint8(255 * cmap)
        #map_np = cv2.cvtColor(map_np, cv2.COLOR_RGB2BGR) # RGB
        #heatmap_np = cv2.applyColorMap(cmap, cv2.COLORMAP_JET) # COLORMAP has to be applie to BGR
        #map_np = cv2.addWeighted(heatmap_np, 0.5, im_np, 1, 0)  
        #map_np = cv2.cvtColor(map_np, cv2.COLOR_BGR2RGB) # RGB
        map_pil = Image.fromarray(map_np)
        return map_pil, map_np

    def save(self, result_folder_path, im_np):
        # draw on images
        # im_pil = self.crop_image(im_pil)
        #im_np = np.array(im_pil)
        #im_np = cv2.cvtColor(im_np, cv2.COLOR_BGR2RGB)
        #cv2.putText(im_np, classes[val] + ": %.2f" % (prob), (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1,cv2.LINE_AA)
        # cv2.imwrite(os.path.join( result_folder_path, f"{classes[val]}_{name}") , im_np)
        # cv2.imwrite(os.path.join( result_folder_path, f"{name.replace('.png', '_heatmap.png')}") , im_np)
        cv2.imwrite(os.path.join(result_folder_path, f"{name}"), im_np)


if __name__ == "__main__":
    classes = ['not_ok', 'ok']
    net_size = (3, 224, 224)
    net_path = f"./models/resnet50_crop.pt"
    device = "cuda:0"
    # device = "cpu"
    test_folder_path = os.path.join("data", "dataset-crop", "val")
    result_folder_path = os.path.join("data", "dataset-crop", "results")

    cl = ClassifierAct(net_path, device, net_size)
    names = os.listdir(test_folder_path)
    for name in names:
        filepath = os.path.join(test_folder_path, name)

        start = timeit.default_timer()
        im_pil = Image.open(filepath).convert("RGB")
        #im_pil = im_pil.rotate(-90, expand=True)  # TODO issue with image rotation
        im_np = np.array(im_pil)
    
        out = cl.predict(im_pil)
        
        # use softmax to retreive probabilities
        prob = torch.max( torch.softmax(out, dim=1).reshape(-1) ).item()
        val = out.argmax(dim=-1, keepdim=True).item()

        # Get activation map
        map_pil, map_np = cl.get_activation_map(im_pil,out) # RGB 
 
        map_np = cv2.cvtColor(map_np, cv2.COLOR_RGB2BGR) # RGB
        cv2.putText(map_np, classes[val] + ": %.2f" % (prob),(280,280), cv2.FONT_HERSHEY_SIMPLEX, 4.2,(255,0,0),10,cv2.LINE_AA)  
        #cv2.namedWindow("map", cv2.WINDOW_NORMAL)   
        #cv2.imshow('im', im_np)
        #cv2.imshow('map', map_np)
        #cv2.waitKey(0)

        stop = timeit.default_timer()
        print(name + " " + str(int((stop - start) * 1000)) + " ms" + " prob: " + str(prob) + "class: ", classes[val])

        cv2.imwrite(os.path.join(result_folder_path, f"{name}"), map_np)
        #cl.save(result_folder_path, im_pil, val, prob)