import os
import sys
import argparse
import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import sampler

import torchvision.transforms as T



class MnistDataset(Dataset):

    def __init__(self, pathToCsv, transforms=None):
        super().__init__()

        self.pathToCsv = pathToCsv
        self.transforms = transforms

        df = pd.read_csv(pathToCsv, names=('path', 'label'))

        # I made csv files on windows so its contain paths in windows format
        if sys.platform == "linux" or sys.platform == "linux2":
            for i in range(len(df)):
                path = df['path'][i]
                path = path.replace('\\', '/')
                
                df.at[i,'path'] = path
            
        self.df = df


    def __getitem__(self, index):
        
        row = self.df.loc[index]
        label = row['label']

        path = row['path']
        img = plt.imread(path)

        if self.transforms is not None:
            img = self.transforms(img)

        sample = (img, label, path)

        return sample

    def __len__(self):
        return len(self.df)


class SmallConvNet(nn.Module):

    def __init__(self, image_channels, hidden_channels, num_classes):

        super().__init__()

        self.conv1 = nn.Conv2d(image_channels, hidden_channels[0], kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size = 3, stride = 1, padding = 1)

        self.bn1 = nn.BatchNorm2d(hidden_channels[0])
        self.bn2 = nn.BatchNorm2d(hidden_channels[1])
        self.bn3 = nn.BatchNorm2d(hidden_channels[2])

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(7 * 7 * hidden_channels[2], 100)
        self.fc2 = nn.Linear(100, num_classes)
        self.drop = nn.Dropout(p = 0.5)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x



def parse_args():

    """
    A function for parsing arguments from command line
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", help="set the mode of model, train or inference", type=str)
    parser.add_argument("--dataset", help="path to csv file with train image file paths and class labels", type=str)
    parser.add_argument("--model", help="path for saving and loading model", type=str)
    parser.add_argument("--input", help="path to csv file with test image file paths and class labels", type=str)
    parser.add_argument("--output", help="path for saving csv file with model's predictions", type=str)

    args = parser.parse_args()

    correctInput = True

    if args.mode == 'train':

        if args.dataset is None or not(os.path.isfile(args.dataset)):
            print('Could not find the dataset csv file')
            correctInput = False
        
        if args.model is None:
            print('Model save path is not specified')
            correctInput = False

    elif args.mode == 'inference':

        if args.model is None or not(os.path.isfile(args.model)):
            print('Could not find the model file')
            correctInput = False

        if args.input is None or not(os.path.isfile(args.input)):
            print('Could not find the input csv file')
            correctInput = False

        if args.output is None:
            print('Output save path is not specified')
            correctInput = False

    else:
        print('Mode argument must be train or inference')
        correctInput = False

    return correctInput, args



def load_dataset(mode, pathToCsv):

    """
    A function that loads the mnist dataset and creates batch loaders
    If mode is train then function return train and validation loaders
    If mode is inference then it return test loader
    """
    
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.1307), std=(0.3081))
    ])

    dataSet = MnistDataset(pathToCsv, transforms)


    if mode == 'train':

        num_samples = dataSet.__len__()
        num_valid = 4000
        #num_samples = 640
        #num_valid = 120
        num_train = num_samples - num_valid
        trainLoader = DataLoader(dataSet, batch_size=16, sampler=sampler.SubsetRandomSampler(range(0, num_train)))
        validLoader = DataLoader(dataSet, batch_size=16, sampler=sampler.SubsetRandomSampler(range(num_train, num_samples)))
        
        return trainLoader, validLoader


    elif mode == 'inference':

        testLoader = DataLoader(dataSet, batch_size=16)
        return testLoader




def check_accuracy(model, dataLoader, typeOfCheck, device=torch.device('cpu')):
    """
    Checking accuracy on given dataSet in dataLoader
    """

    model = model.to(device=device)

    typeOfCheck = str.lower(typeOfCheck)

    model.eval()

    num_samples = 0
    num_correct = 0

    if typeOfCheck == 'train':
        num_batches = len(dataLoader) / 10  # Because iterating over whole train set is too expansive
    else:
        num_batches = len(dataLoader)

    with torch.no_grad():
        for i, (x, y, _) in enumerate(dataLoader):
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            _, preds = scores.max(1)

            num_samples += preds.shape[0]
            num_correct += (preds == y).sum()

            if typeOfCheck == 'train' and i == num_batches:
                break

    acc = num_correct / num_samples
    print('Got %d / %d correct (%.3f) in %s'% (num_correct, num_samples, acc, typeOfCheck))
    return acc



def train_model(model, optimizer, trainLoader, validationLoader, savePath, num_epoch=1, device=torch.device('cpu'), printAndSaveEvery=None):
    """
    Function for making model training
    """

    print("Starting training")
    best_acc = 0
    train_accuracies = []
    val_accuracies = []
    start_ep = 0

    counterWithoutImprovements = 0

    model = model.to(device=device)

    for e in range(start_ep, num_epoch):

        if counterWithoutImprovements >= 10:

            oldLr = 0
            for g in optimizer.param_groups:
                oldLr = g['lr']
                g['lr'] = oldLr * 0.1
            
            print('Learning rate reduction from ' + str(oldLr) + ' to ' + str(oldLr * 0.1))
            counterWithoutImprovements = 0


        print("Start " + str(e) + " epoch")

        for t, (x, y, _) in enumerate(trainLoader):

            model.train()
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if printAndSaveEvery is not None and t != 0 and t % printAndSaveEvery == 0:

                print("Iteration " + str(t) + ":")

                train_acc = check_accuracy(model, trainLoader, 'train', device)
                val_acc = check_accuracy(model, validationLoader, 'validation', device)

                train_accuracies.append(train_acc)
                val_accuracies.append(val_acc)

                if best_acc < val_acc:
                    print("Goten new best val accuracy. Save new best model")
                    best_acc = val_acc
                    torch.save({
                        'Model_state_dict': model.state_dict(),
                        'Optimizer_state_dict': optimizer.state_dict(),
                        'Num_epoch': e + 1,
                        'Train_accs': train_accuracies,
                        'Val_accs': val_accuracies
                    }, savePath)
                    counterWithoutImprovements = 0
                else:
                    counterWithoutImprovements += 1
            

        train_acc = check_accuracy(model, trainLoader, 'train', device)
        val_acc = check_accuracy(model, validationLoader, 'validation', device)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if best_acc < val_acc:
              print("Goten new best val accuracy. Save new best model")
              best_acc = val_acc
              torch.save({
                  'Model_state_dict': model.state_dict(),
                  'Optimizer_state_dict': optimizer.state_dict(),
                  'Num_epoch': e + 1,
                  'Train_accs': train_accuracies,
                  'Val_accs': val_accuracies
              }, savePath)
              counterWithoutImprovements = 0
        else:
            counterWithoutImprovements += 1

    return(train_accuracies, val_accuracies)



def inference_model(testLoader, pathToModel, pathToSaveCsv, printEvery=None, device=torch.device('cpu')):

    """
    Makes inference of the model on test set in csv file
    """

    checkpoint = torch.load(pathToModel, map_location=device)

    model = SmallConvNet(1, [64, 128, 256], 10)
    model.load_state_dict(checkpoint['Model_state_dict'])
    model.to(device=device)
    
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    optimizer.load_state_dict(checkpoint['Optimizer_state_dict'])

    model.eval()

    with torch.no_grad():

        with open(pathToSaveCsv, "w", newline='') as csvFile:
            writer = csv.writer(csvFile, delimiter=',')

            for i, (x, y, paths) in enumerate(testLoader):

                if printEvery is not None and i % printEvery == 0:
                    print('%d examples was predicted'%(i * 16))

                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.long)

                scores = model(x)
                _, preds = scores.max(1)

                for path, pred in zip(paths, preds):

                    writer.writerow([path, pred.item()])




if __name__ == '__main__':


    correctInput, args = parse_args()

    if not correctInput:
        quit()


    USE_GPU = True

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('Using ' + device.type + ' for model computations')


    if args.mode == 'train':

        trainLoader, validLoader = load_dataset(args.mode, args.dataset)

        model = SmallConvNet(1, [64, 128, 256], 10)
        optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)

        train_model(model, optimizer, trainLoader, validLoader, args.model, num_epoch=3, device=device, printAndSaveEvery=200)

        

    elif args.mode == 'inference':

        testLoader = load_dataset(args.mode, args.input)

        inference_model(testLoader, args.model, args.output, 200, device)
