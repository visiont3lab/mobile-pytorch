import torch 
import torchvision

classes = {
    "Abyssinian": 0,  "basset": 1, "Bengal": 2, "Bombay": 3, "British":4, "Egyptian": 5,
    "german": 6, "havanese": 7, "keeshond": 8, "Maine": 9, "newfoundland": 10,
    "pomeranian": 11, "Ragdoll": 12, "saint": 13, "scottish": 14, "Siamese": 15,
    "staffordshire": 16, "yorkshire": 17, "american": 18, "beagle": 19, "Birman": 20,
    "boxer": 21, "chihuahua": 22, "english": 23, "great": 24, "japanese": 25,
    "leonberger": 26, "miniature": 27, "Persian": 28, "pug": 29, "Russian": 30,
    "samoyed": 31, "shiba": 32, "Sphynx": 33, "wheaten": 34
}


def load_model(pretrained, num_classes):

    class FinalLayer(torch.nn.Module):
        def __init__(self, in_features, num_classes):
            super(FinalLayer, self).__init__()
            self.fc1 = torch.nn.Linear(in_features=in_features, out_features=100)
            self.fc2 = torch.nn.Linear(in_features=100, out_features=num_classes)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            x = torch.log_softmax(x, dim=-1)
            return x

    # selected net
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

    if pretrained:
        # La batch nomrmalization serve a normalizzare l'output dei neuroni in modo
        # che il gradiente sia sempre alto. Questi parametetri (mean, std) vanno stimati
        for name, param in net.named_parameters():
            if ("bn" not in name):
                param.requires_grad = False
            
        net.fc = FinalLayer(net.fc.in_features, num_classes)
    
    return net

