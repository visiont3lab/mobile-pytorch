import torch 
import torchvision

# Docs
# Pretrained Models: https://pytorch.org/vision/stable/models.html

classes = {
    "not_ok": 0,  "ok": 1
}


def load_model_efficient(pretrained, num_classes):
    # selected net
    # net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # net = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    # net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    net = torchvision.models.efficientnet_b7(weights=torchvision.models.EfficientNet_B7_Weights.DEFAULT)

    if pretrained:
        # La batch nomrmalization serve a normalizzare l'output dei neuroni in modo
        # che il gradiente sia sempre alto. Questi parametetri (mean, std) vanno stimati
        for name, param in net.named_parameters():
            if ("bn" not in name):
                param.requires_grad = False

    # net.fc = FinalLayer(net.fc.in_features, num_classes)
    # net.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    net.classifier[1] = torch.nn.Linear(in_features=2560, out_features=num_classes)
    return net


def load_model(pretrained, num_classes):

    class FinalLayer(torch.nn.Module):
        def __init__(self, in_features, num_classes):
            super(FinalLayer, self).__init__()
            self.fc1 = torch.nn.Linear(in_features=in_features, out_features=100)
            self.fc2 = torch.nn.Linear(in_features=100, out_features=num_classes)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            #x = torch.log_softmax(x, dim=-1)
            #x = torch.softmax(x, dim=-1)
            return x

    # selected net
    #net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    #net = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.DEFAULT)
    net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    #net = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)

    if pretrained:
        # La batch nomrmalization serve a normalizzare l'output dei neuroni in modo
        # che il gradiente sia sempre alto. Questi parametetri (mean, std) vanno stimati
        for name, param in net.named_parameters():
            if ("bn" not in name):
                param.requires_grad = False
            
    net.fc = FinalLayer(net.fc.in_features, num_classes)
    
    return net


if __name__ == "__main__":
    net = load_model(pretrained=True, num_classes=2)
    print(net)
