import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import dataloader
from torchvision.transforms.functional import pad

batch_size=256
learning_rate=0.0002
num_epoch=10

mnist_train=dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test=dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)
train_loader=torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
test_loader=torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(1,16,5),
            nn.ReLU(),
            nn.Conv2d(16,32,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc_layer=nn.Sequential(
            nn.Linear(64*3*3, 100),
            nn.ReLU(),
            nn.Linear(100,10)
        )
    
    def forward(self, x):
        out=self.layer(x)
        out=out.view(batch_size, -1)
        out=self.fc_layer(out)
        return out

def conv_2_block(in_dim, out_dim):
    model=nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

def conv_3_block(in_dim, out_dim):
    model=nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

class VGG(nn.Module):
    def __init__(self, base_dim, num_classes=2):
        super(VGG, self).__init__()
        self.feature=nn.Sequential(
            conv_2_block(3, base_dim),
            conv_2_block(base_dim, 2*base_dim),
            conv_3_block(2*base_dim, 4*base_dim),
            conv_3_block(4*base_dim, 8*base_dim),
            conv_3_block(8*base_dim, 8*base_dim),
        )
        self.fc_layer=nn.Sequential(
            nn.Linear(8*base_dim*7*7, 100),
            nn.ReLU(True),
            nn.Linear(100,20),
            nn.ReLU(True),
            nn.Linear(20,num_classes),
        )
    
    def forward(self, x):
        x=self.feature(x)
        x=x.view(x.size(0),-1)
        x=self.fc_layer(x)
        return x

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=CNN().to(device)
loss_func=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_arr=[]
for i in range(num_epoch):
    for j, [image, label] in enumerate(train_loader):
        x=image.to(device)
        y_=label.to(device)

        optimizer.zero_grad()
        output=model.forward(x)
        loss=loss_func(output,y_)
        loss.backward()
        optimizer.step()

        if j%100==0:
            print(loss)
            loss_arr.append(loss.cpu().detach().numpy())

correct=0
total=0

with torch.no_grad():
    for image, label in test_loader:
        x=image.to(device)
        y_=label.to(device)

        output=model.forward(x)
        _, output_index=torch.max(output, 1)

        total+=label.size(0)
        correct+=(output_index==y_).sum().float()

    print("Accuracy of Test Data: {}".format(100*correct/total))