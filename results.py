# Setup GPU Device
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

path2weights = "/content/drive/MyDrive/T3lab/best_model.pt"
weights = torch.load(path2weights)
net.load_state_dict(weights)

# Send model to device
net.to(device)

# Tell the model layer that we are going to use the model in evaluation  mode!
net.eval()

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=10, shuffle=True)

# Predict Classication
cm = np.zeros((num_classes,num_classes))
names_pred = [ "Pred: " + n for n in classes]

y_pred = []
y_true = []

with torch.no_grad():
    for x,y in train_dl:
        x = x.to(device)
        y_hat = net.forward(x).argmax(dim=-1,keepdim=True).cpu().numpy().reshape(-1)
        y_pred.extend(y_hat)
        y_true.extend(y.cpu().numpy())

# Visualize results
cm += confusion_matrix(y_true, y_pred)
print("Confusion Matrix")
df = pd.DataFrame(cm, columns=names_pred, index=classes)
display(df)