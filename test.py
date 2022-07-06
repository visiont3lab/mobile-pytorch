import torch
from PIL import Image
from torchvision import transforms
from model import load_model, classes

path_weights = "models/net.pt"
net = load_model(pretrained=False, num_classes=len(classes))
weights = torch.load(path_weights)
net.load_state_dict(weights)
net.eval()

net_mobile = torch.jit.load("models/mobile_net.ptl")

def run(filename, model, size):

    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    im_tensor = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    with torch.no_grad():
        log_prob = model.forward(im_tensor) # 1xCxHxW
        val = log_prob.argmax(dim=-1,keepdim=True).item()
        print(f"Class: {val}")
        print(f"Prob:  {torch.exp(log_prob)}")
        #prob = log_prob[0, val]

# Compare the models
filename = "data/images/val/Abyssinian/Abyssinian_4.jpg"
run(filename, net_mobile, size=224)
run(filename, net, size=224)
