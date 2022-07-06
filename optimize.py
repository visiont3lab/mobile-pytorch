import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from model import load_model, classes


path_weights = "models/net.pt"
net = load_model(pretrained=False, num_classes=len(classes))
weights = torch.load(path_weights)
net.load_state_dict(weights)
net.eval()

scripted_module = torch.jit.script(net)
# Export full jit version model (not compatible mobile interpreter), leave it here for comparison
#scripted_module.save("net.pt")
# Export mobile interpreter version model (compatible with mobile interpreter)
optimized_scripted_module = optimize_for_mobile(scripted_module)
optimized_scripted_module._save_for_lite_interpreter("models/mobile_net.ptl")