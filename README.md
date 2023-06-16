# ERA V1 Session 7

## Create and Train a Neural Network in Python

An implementation to create and train a simple neural network in python.

## Usage
### model.py

- First we have to import all the neccessary libraries.

```ruby
import torch.nn as nn
import torch.nn.functional as F
```
- Next we build a simple Neural Network.
For this, we define 3 classes **class Model1()**, **class Model2()**, **class Model3()** and pass **nn.Module** as the parameter.

```ruby
class Model1(nn.Module):
```

- Create two functions inside the class to get our model ready. First is the **init()** and the second is the **forward()**.
- We need to instantiate the class for training the dataset. When we instantiate the class, the forward() function will get executed.

### **class Model1()**

#### Target:
    1. Reducing the total parameter count.
#### Results:
    1. Parameters: 4,312
    2. Best Train Accuracy: 88.88 (15th Epoch)
    3. Best Test Accuracy: 98.53 (15th Epoch)
#### Analysis:
    1. I have tried using learning rate 0.01
    2. I used StepLR scheduler.

```ruby
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        )  # 28 >> 26

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        )  # 26 >> 24

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )  # 24 >> 24

        self.pool1 = nn.MaxPool2d(2, 2)  # 24 >> 12

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        )  # 12 >> 10

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        )  # 10 >> 10

        self.pool2 = nn.MaxPool2d(2, 2)  # 10 >> 5

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        )  # 5 >> 5

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        )  # output_size = 1

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.pool2(x)
        x = self.convblock6(x)
        x = self.gap(x)
        x = self.convblock7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
 ```

#### Model1 Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
              ReLU-2            [-1, 8, 26, 26]               0
       BatchNorm2d-3            [-1, 8, 26, 26]              16
           Dropout-4            [-1, 8, 26, 26]               0
            Conv2d-5           [-1, 16, 24, 24]           1,152
              ReLU-6           [-1, 16, 24, 24]               0
       BatchNorm2d-7           [-1, 16, 24, 24]              32
           Dropout-8           [-1, 16, 24, 24]               0
            Conv2d-9           [-1, 10, 24, 24]             160
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11           [-1, 10, 10, 10]             900
             ReLU-12           [-1, 10, 10, 10]               0
      BatchNorm2d-13           [-1, 10, 10, 10]              20
          Dropout-14           [-1, 10, 10, 10]               0
           Conv2d-15           [-1, 10, 10, 10]             900
             ReLU-16           [-1, 10, 10, 10]               0
      BatchNorm2d-17           [-1, 10, 10, 10]              20
          Dropout-18           [-1, 10, 10, 10]               0
        MaxPool2d-19             [-1, 10, 5, 5]               0
           Conv2d-20             [-1, 10, 5, 5]             900
             ReLU-21             [-1, 10, 5, 5]               0
      BatchNorm2d-22             [-1, 10, 5, 5]              20
          Dropout-23             [-1, 10, 5, 5]               0
        AvgPool2d-24             [-1, 10, 1, 1]               0
           Conv2d-25             [-1, 10, 1, 1]             100
      BatchNorm2d-26             [-1, 10, 1, 1]              20
             ReLU-27             [-1, 10, 1, 1]               0
          Dropout-28             [-1, 10, 1, 1]               0
================================================================
Total params: 4,312
Trainable params: 4,312
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.57
Params size (MB): 0.02
Estimated Total Size (MB): 0.59
----------------------------------------------------------------
```

### **class Model2()**

#### Target:
    1. Reducing the total parameter count and increasing test accuracy.
#### Results:
    1. Parameters: 7,524
    2. Best Train Accuracy: 99.50 (15th Epoch)
    3. Best Test Accuracy: 99.11 (11th Epoch)
#### Analysis:
    1. I have tried using learning rate 0.01
    2. I used StepLR scheduler.
    3. I changed convolution blocks and transition blocks.


```ruby
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )  # 28 >> 26



        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )  # 26 >> 24



        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )  # 24 >> 24



        self.pool1 = nn.MaxPool2d(2, 2)  # 24 >> 12



        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )  # 12 >> 10



        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )  # 10 >> 8



        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )  # 8 >> 6



        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )  # 6 >> 4



        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )  # 4 >> 2



        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(2, 2), padding=0, bias=False),
        )  # 2 >> 1



    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
 ```

#### Model2 Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
              ReLU-2            [-1, 8, 26, 26]               0
       BatchNorm2d-3            [-1, 8, 26, 26]              16
            Conv2d-4           [-1, 16, 24, 24]           1,152
              ReLU-5           [-1, 16, 24, 24]               0
       BatchNorm2d-6           [-1, 16, 24, 24]              32
            Conv2d-7           [-1, 10, 24, 24]             160
         MaxPool2d-8           [-1, 10, 12, 12]               0
            Conv2d-9           [-1, 10, 10, 10]             900
             ReLU-10           [-1, 10, 10, 10]               0
      BatchNorm2d-11           [-1, 10, 10, 10]              20
           Conv2d-12             [-1, 16, 8, 8]           1,440
             ReLU-13             [-1, 16, 8, 8]               0
      BatchNorm2d-14             [-1, 16, 8, 8]              32
           Conv2d-15             [-1, 10, 6, 6]           1,440
             ReLU-16             [-1, 10, 6, 6]               0
      BatchNorm2d-17             [-1, 10, 6, 6]              20
           Conv2d-18             [-1, 10, 4, 4]             900
             ReLU-19             [-1, 10, 4, 4]               0
      BatchNorm2d-20             [-1, 10, 4, 4]              20
           Conv2d-21             [-1, 10, 2, 2]             900
             ReLU-22             [-1, 10, 2, 2]               0
      BatchNorm2d-23             [-1, 10, 2, 2]              20
           Conv2d-24             [-1, 10, 1, 1]             400
================================================================
Total params: 7,524
Trainable params: 7,524
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.45
Params size (MB): 0.03
Estimated Total Size (MB): 0.48
----------------------------------------------------------------
```

### **class Model3()**

#### Target:
    1. Reducing the total parameter count and increasing test accuracy.
#### Results:
    1. Parameters: 7,524
    2. Best Train Accuracy: 99.50 (15th Epoch)
    3. Best Test Accuracy: 99.28 (14th Epoch)
#### Analysis:
    1. I have tried using learning rate 0.01
    2. I used ReduceLROnPlateau scheduler.
    3. I changed convolution blocks and transition blocks.


```ruby
class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )  # 28 >> 26



        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )  # 26 >> 24



        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )  # 24 >> 24



        self.pool1 = nn.MaxPool2d(2, 2)  # 24 >> 12



        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )  # 12 >> 10



        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )  # 10 >> 8



        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )  # 8 >> 6



        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )  # 6 >> 4



        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )  # 4 >> 2



        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(2, 2), padding=0, bias=False),
        )  # 2 >> 1



    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
 ```
#### Model3 Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
              ReLU-2            [-1, 8, 26, 26]               0
       BatchNorm2d-3            [-1, 8, 26, 26]              16
            Conv2d-4           [-1, 16, 24, 24]           1,152
              ReLU-5           [-1, 16, 24, 24]               0
       BatchNorm2d-6           [-1, 16, 24, 24]              32
            Conv2d-7           [-1, 10, 24, 24]             160
         MaxPool2d-8           [-1, 10, 12, 12]               0
            Conv2d-9           [-1, 10, 10, 10]             900
             ReLU-10           [-1, 10, 10, 10]               0
      BatchNorm2d-11           [-1, 10, 10, 10]              20
           Conv2d-12             [-1, 16, 8, 8]           1,440
             ReLU-13             [-1, 16, 8, 8]               0
      BatchNorm2d-14             [-1, 16, 8, 8]              32
           Conv2d-15             [-1, 10, 6, 6]           1,440
             ReLU-16             [-1, 10, 6, 6]               0
      BatchNorm2d-17             [-1, 10, 6, 6]              20
           Conv2d-18             [-1, 10, 4, 4]             900
             ReLU-19             [-1, 10, 4, 4]               0
      BatchNorm2d-20             [-1, 10, 4, 4]              20
           Conv2d-21             [-1, 10, 2, 2]             900
             ReLU-22             [-1, 10, 2, 2]               0
      BatchNorm2d-23             [-1, 10, 2, 2]              20
           Conv2d-24             [-1, 10, 1, 1]             400
================================================================
Total params: 7,524
Trainable params: 7,524
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.45
Params size (MB): 0.03
Estimated Total Size (MB): 0.48
----------------------------------------------------------------
```

 
### utils.py
- In this we created two functions **train()** and **test()**
- train() funtion computes the prediction, traininng accuracy and loss

```ruby
def train(model, device, train_loader, optimizer, criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    
    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))
```

- And test() function calculates the loss and accuracy of the model

```ruby
def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```

### S7.ipynb
- First we have to load MNIST dataset then we have to create train and test data
```ruby
from torchvision import datasets

train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)
```


- Plotting the dataset of **train_loader**


![train_loader](https://github.com/GunaKoppula/Neural-Networks/assets/61241928/e15fdb8e-f44b-4a4c-80d0-0128491ea760)


- **Training and Testing trigger**
```ruby
model = Model3().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    factor=0.1, patience=10, threshold=0.0001, threshold_mode='abs')


for epoch in range(0, 15):
    print(f"EPOCH: {epoch+1}")
    utils.train(model, device, train_loader, optimizer, epoch)
    utils.test(model, device, test_loader)
```

I used total 15 epoch
```
EPOCH: 13
Loss=0.0234049204736948 Batch_id=468 Accuracy=99.49: 100%|██████████| 469/469 [00:20<00:00, 23.39it/s]

Test set: Average loss: 0.0265, Accuracy: 9917/10000 (99.17%)
```

