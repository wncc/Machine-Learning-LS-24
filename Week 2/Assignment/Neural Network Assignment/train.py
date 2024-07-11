#imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io


class SmpDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)
        return (image, y_label)


# Transform to resize images to 64x64 and convert to tensor
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x) ,dim=1)
        return x


#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameter
input_size = 3 * 64 * 64
num_classes = 2
learning_rate = 0.0001
batch_size = 32
num_epochs = 200

#load data
dataset = SmpDataset(csv_file='dataset.csv', root_dir='imgs',
                     transform=transform)

train_set, test_set = torch.utils.data.random_split(dataset , [int(len(dataset) * 0.9) , len(dataset) - int(len(dataset) * 0.9)]  )

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

#initialize the network
model= NN(input_size=input_size,num_classes=num_classes,).to(device=device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters() , lr =learning_rate)

#train network

for epochs in range(num_epochs):
    curr_loss = 0
    for batch_idx , (data , targets) in enumerate(train_loader):
        #get to cuda
        data = data.to(device=device)
        targets = targets.to(device=device)

        #reshape the data
        data = data.reshape(data.shape[0] , -1)

        #forward
        scores = model(data)
        loss = criterion(scores, targets)
        curr_loss = loss

        #backwards
        optimizer.zero_grad()
        loss.backward()

        #gradient descent step
        optimizer.step()
    if ( epochs % 10 == 0):
        print(f"current epoch {epochs} with loss {curr_loss.item():.4f}")

#check accuray now
def check_accuracy(loader , model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x , y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0] , -1 )

            scores = model(x)
            _,predictions = scores.max(1)
            num_correct += (predictions == y ).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {(float(num_correct)*100/float(num_samples)):.2f}")
    model.train()

check_accuracy(train_loader , model)
check_accuracy(test_loader, model)


