import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim

import time
import numpy as np
import glob
from PIL import Image


class test(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, stride = 1, padding_mode = 'replicate')
        self.conv2 = nn.Conv2d(64, 64, 5, stride = 1, padding_mode = 'replicate')
        self.conv3 = nn.Conv2d(64, 64 ,3, stride = 1, padding_mode = 'replicate')
        self.conv4 = nn.Conv2d(64, 64, 3, stride = 1, padding_mode = 'replicate')
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.batchnorm4 = nn.BatchNorm2d(64)
        # 52* 52* 64
        self.fc1 = nn.Linear(52*52*64, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 6)
        self.batchnorm5 = nn.BatchNorm1d(128)
        self.batchnorm6 = nn.BatchNorm1d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, 52*52*64)
        x = self.fc1(x)
        x = self.batchnorm5(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.batchnorm6(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class pretrain_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = torchvision.models.vgg16(pretrained=True).features
        for param in self.pretrained.parameters():
            param.requires_grad = False
        self.fc1 = nn.Linear(7*7*512 ,512)
        self.fc2 = nn.Linear(512 ,6)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pretrained(x)
        x = x.view(-1, 7*7*512)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class data(Dataset):
    def __init__(self, txt_file, root_dir, transform = None):
        with open(txt_file, 'r') as f:
            self.label = f.readlines()
        self.root = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        files = sorted(glob.glob(self.root + '/*.png'))
        image = Image.open(files[idx])

        # delete alpha channel
        #if(image.shape[2] == 4):
        #    image = image[:,:,0:3]
        
        label = [float(x) for x in self.label[idx][:-1].split(' ')]
        label = np.array(label)
        label = torch.from_numpy(label)
        if self.transform:
            image = self.transform(image)
        
        return image, label


class data_map(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx][0]), self.dataset[idx][1]


transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


dataset = data('/home/jeff/masks/mask_face/label.txt', '/home/jeff/masks/mask_face')

tranin_dataset, test_dataset = torch.utils.data.random_split(dataset, [4000, 800])

tranin_dataset = data_map(tranin_dataset, transform_train)
test_dataset = data_map(test_dataset, transform_test)

train_loader = torch.utils.data.DataLoader(tranin_dataset, batch_size = 32, shuffle = True, num_workers = 4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle = True, num_workers = 4, pin_memory=True)

#net = test()
net = pretrain_model()
if torch.cuda.is_available():
    net.cuda()

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters())

def train(epochs, train_loader, test_loader, model, optimizer, criterion, save_path, earlystop_patience = None, min_change = 0.001):
    validate_best = np.Inf
    epoch_change = 0
    epoch_not_change = 0

    for i in range(epochs):
        start = time.time()
        train_loss = 0.0
        test_loss = 0.0
        model.train()
        for (image, key_pts) in train_loader:

            #image = image.resize_(32, 3, 224, 224)
            optimizer.zero_grad()

            if torch.cuda.is_available():
                image, key_pts = image.cuda(), key_pts.cuda()

            output = model.forward(image)
            loss = criterion(output, key_pts)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        for (image, key_pts) in test_loader:

            if torch.cuda.is_available():
                image, key_pts = image.cuda(), key_pts.cuda()
            
            output = model.forward(image)
            loss = criterion(output, key_pts)

            test_loss += loss.item()

        print('epoch: {}, train_loss: {:.6f}, test_loss: {:.6f}, elapsed_time: {:.6f}'.format(i+1, train_loss/len(train_loader), test_loss/len(test_loader), time.time() - start))
        
        if type(earlystop_patience) != int:
            if test_loss/len(test_loader) < validate_best:
                validate_best = test_loss/len(test_loader)
                torch.save(model.state_dict(), save_path)
                print('model save')
        else:
            if (validate_best - test_loss/len(test_loader)) > min_change:
                validate_best = test_loss/len(test_loader)
                epoch_change = i
                torch.save(model.state_dict(), save_path)
                print('model save')
            else:
                epoch_not_change = i
            if epoch_not_change - epoch_change > earlystop_patience:
                print('earlystopping')
                break
            

train(10, train_loader, test_loader, net, optimizer, criterion, '/home/jeff/tett.pth', 2, min_change = 0.01)

model = torch.load('/home/jeff/tett.pth')
if torch.cuda.is_available():
    model.cuda()
model.eval()

image = Image.open('/home/jeff/Downloads/self-built-masked-face-recognition-dataset/AFDB_face_dataset/aidai/1_0_aidai_0001.jpg')
image = transform_test(image)
image = image.unsqueeze(0)
image = image.cuda()
output = model(image)
output = output.cpu()
output = output.numpy()
print(output)
