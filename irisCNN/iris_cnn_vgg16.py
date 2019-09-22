import os
from PIL import Image
from torchsummary import summary
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import numpy as np
import sys

class CustomImageDataset(Dataset):
  def read_data_set(self):
    all_img_files = []
    all_labels = []
    
    class_names = os.walk(self.data_set_path).__next__()[1]
    class_names.sort()
    
    for index, class_name in enumerate(class_names):
      print(index)      
      label = index
      img_dir = os.path.join(self.data_set_path, class_name)
      img_files = os.walk(img_dir).__next__()[2]
      
      for img_file in img_files:
        img_file = os.path.join(img_dir, img_file)
        img = Image.open(img_file)
        if img is not None:
          all_img_files.append(img_file)
          all_labels.append(label)

    return all_img_files, all_labels, len(all_img_files), len(class_names)

  def __init__(self, data_set_path, transforms=None):
    self.data_set_path = data_set_path
    self.image_files_path, self.labels, self.length, self.num_classes = self.read_data_set()
    self.transforms = transforms
  
  def __getitem__(self, index):
    image = Image.open(self.image_files_path[index])
    image = image.convert("RGB")

    if self.transforms is not None:
      image = self.transforms(image)

    return {'image': image, 'label': self.labels[index]}

  def __len__(self):
    return self.length


class CustomConvNet(nn.Module):
  def __init__(self, num_classes):
    super(CustomConvNet, self).__init__()

    self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1), nn.ReLU())
    self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
    self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1), nn.ReLU())
    self.conv4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
    self.conv5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1), nn.ReLU())
    self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.ReLU())
    self.conv7 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
    self.conv8 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1), nn.ReLU())
    self.conv9 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.ReLU())
    self.conv10 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
    self.conv11 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.ReLU())
    self.conv12 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.ReLU())
    self.conv13 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
    self.fc1 = nn.Sequential(nn.Linear(11*2*512, 4096),  nn.ReLU(), nn.Dropout(), nn.BatchNorm1d(4096))
    self.fc2 = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(), nn.BatchNorm1d(4096))
    self.fc3 = nn.Sequential(nn.Linear(4096, num_classes))

  def forward(self, x):
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.conv5(out)
    out = self.conv6(out)
    out = self.conv7(out)
    out = self.conv8(out)
    out = self.conv9(out)
    out = self.conv10(out)
    out = self.conv11(out)
    out = self.conv12(out)
    out = self.conv13(out)
    out = out.view(-1, 11*2*512)
    out = self.fc1(out)
    out = self.fc2(out)
    out = self.fc3(out)
    return out
  
hyper_param_epoch = 80
hyper_param_batch = 50
hyper_param_learning_rate = 0.001

transforms_train = transforms.Compose([transforms.Resize((360, 80)),
                                       transforms.RandomRotation(10.),
                                       transforms.ToTensor()])

transforms_test = transforms.Compose([transforms.Resize((360, 80)),
                                      transforms.ToTensor()])

train_data_set = CustomImageDataset(data_set_path="/content/gdrive/My Drive/iris/train", transforms=transforms_train)
train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch, shuffle=True)

test_data_set = CustomImageDataset(data_set_path="/content/gdrive/My Drive/iris/test", transforms=transforms_test)
test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch, shuffle=True)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_classes = train_data_set.num_classes
custom_model = CustomConvNet(num_classes=num_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(custom_model.parameters(), lr=hyper_param_learning_rate)


for e in range(hyper_param_epoch):
  for i_batch, item in enumerate(train_loader):
    images = item['image'].to(device)
    labels = item['label'].to(device)
    
    outputs = custom_model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print('Epoch [{}/{}], Loss: {:.4f}'.format(e + 1, hyper_param_epoch, loss.item()))

custom_model.eval()

'''
np.set_printoptions(threshold=np.inf)
with open('/content/gdrive/My Drive/iris/some_file.txt', 'w') as f:
  for name, param in custom_model.named_parameters():
    if param.requires_grad:
      print(param.data.shape)
      f.write(name + " = ")
      f.write("[")
      for p in param.data:
        f.write(str(p.cpu().numpy()))
        f.write(",")
      f.write("]")
      f.write('\n')
'''
summary(custom_model, (3, 360, 80))

with torch.no_grad():
  correct = 0
  total = 0
  for item in test_loader:
    images = item['image'].to(device)
    labels = item['label'].to(device)

    outputs = custom_model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += len(labels)
    print('predicted : ',predicted, '\nlabels : ',labels)
    correct += (predicted == labels).sum().item()
  print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))