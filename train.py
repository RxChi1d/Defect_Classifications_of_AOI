import os
import numpy as np
from tqdm import tqdm

import torch
import argparse
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter

# Import models
from model.efficientnet_b6 import myModel as efficientnet_b6
from model.efficientnet_b7 import myModel as efficientnet_b7
from model.efficientnet_v2_l import myModel as efficientnet_v2_l
from model.efficientnet_v2_m import myModel as efficientnet_v2_m
from model.efficientnet_v2_s import myModel as efficientnet_v2_s

def same_seed(seed): 
    # Fixes random number generator seeds for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def train(train_loader, model, criterion, optimizer, writer, epoch, device):
    model.train()
    loss_record, correct, count = [], 0, 0
    for data, target in train_loader:
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)      
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        loss_record.append(loss.detach().item())

        # calculate the nums of correct output
        pred = torch.max(output, 1)[1]
        correct += (pred == target).sum()
        count += data.size()[0]

    mean_train_loss = sum(loss_record)/len(loss_record)
    mean_train_acc = correct/count * 100.
    writer.add_scalar('Loss/train', mean_train_loss, epoch)
    writer.add_scalar('Acc/train', mean_train_acc, epoch)
    return mean_train_loss, mean_train_acc

def valid(valid_loader, model, criterion, writer, epoch, device):
    model.eval()
    loss_record, correct, count = [], 0, 0

    for data, target in valid_loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)

        loss_record.append(loss.item())

        pred = torch.max(output, 1)[1]
        correct += (pred == target).sum()
        count += data.size()[0]

    mean_valid_loss = sum(loss_record)/len(loss_record)
    mean_valid_acc = correct/count * 100.
    writer.add_scalar('Loss/valid', mean_valid_loss, epoch)
    writer.add_scalar('Acc/valid', mean_valid_acc, epoch)
    return mean_valid_loss, mean_valid_acc

def train_loop(config, device):
    # Load the data
    train_data = ImageFolder(config['train_folder'],transform=transform_train)
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, pin_memory=True)  
    valid_data = ImageFolder(config['valid_folder'],transform=transform_test)
    valid_loader = DataLoader(valid_data, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    
    criterion = nn.CrossEntropyLoss()
    
    # Load the model
    model_name = config['model_name']
    if model_name == 'efficientnet_b6':
        model = efficientnet_b6().to(device)
    elif model_name == 'efficientnet_b7':
        model = efficientnet_b7().to(device)
    elif model_name == 'efficientnet_v2_l':
        model = efficientnet_v2_l().to(device)
    elif model_name == 'efficientnet_v2_m':
        model = efficientnet_v2_m().to(device)
    elif model_name == 'efficientnet_v2_s':
        model = efficientnet_v2_s().to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Writer of tensoboard.
    writer = SummaryWriter('runs/without_kfold')
    
    # Create directory of saving models.
    if not os.path.isdir('./weight'):    
        os.mkdir('./weight') 
        
    # Ceate folder of log
    if not os.path.isdir(r'./log'):
        os.mkdir(r'./log')
    
    n_epochs, best_loss = config['n_epochs'], np.inf
    train_pbar = tqdm(range(1,n_epochs+1), position=0, leave=True)
    for epoch in train_pbar:        
        train(train_loader, model, criterion, optimizer, writer, epoch, device)
        mean_valid_loss, mean_valid_acc = valid(valid_loader, model, criterion, writer, epoch, device)
        
        # Log the best epoch
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            best_log = f"Epoch: {epoch}, acc: {mean_valid_acc}, loss: {mean_valid_loss}"
            
            # Save best model
            torch.save(model.state_dict(), './weight/{}.ckpt'.format(config['model_name']))
        
        # Show the imformation on tqdm
        train_pbar.set_description(f'Epoch: [{epoch}/{n_epochs}]')
        train_pbar.set_postfix({'loss': mean_valid_loss})
        
    # Save the best log
    with open(r'./log/best_epoch.txt','a') as f:
            f.write(best_log+'\n')
            
    return best_log

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_models", help="show the models which can be trained", action="store_true")
    parser.add_argument("-s", "--seed", help="set a seed", type=int, default=1314520)
    parser.add_argument("-e", "--n_epochs", help="set the number of epoch", type=int, default=1000)
    parser.add_argument("-b", "--batch_size", help="set the batch size", type=int, default=128)
    parser.add_argument("-lr", "--learning_rate", help="set the learning rate", type=float, default=1e-4)
    parser.add_argument("-td", "--train_folder", help="the path of train data folder", type=str, default=r'./aoi_data/labels/train_data')
    parser.add_argument("-vd", "--valid_folder", help="the path of valid data folder", type=str, default=r'./aoi_data/labels/valid_data')
    parser.add_argument("-m", "--model_name", help="choose the model to train", type=str, default=r'efficientnet_v2_m')
    args = parser.parse_args()
    
    if args.show_models:
        print(['efficientnet_b6','efficientnet_b7','efficientnet_v2_l','efficientnet_v2_m','efficientnet_v2_s'])
    else:
        # Transform
        transform_train = transforms.Compose([
            transforms.Resize(128),
            transforms.GaussianBlur(5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.6951, 0.6951, 0.6951],
                std=[0.1011, 0.1011, 0.1011]
            )
        ])

        transform_test = transforms.Compose([
            transforms.Resize(128),
            transforms.GaussianBlur(5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.6951, 0.6951, 0.6951],
                std=[0.1011, 0.1011, 0.1011]
            )
        ])
        
        config = {
            'seed': args.seed,
            'n_epochs': args.n_epochs,
            'batch_size': args.batch_size, 
            'learning_rate': args.learning_rate,
            'train_folder': args.train_folder,
            'valid_folder': args.valid_folder,
            'model_name': args.model_name
        }

        # Set seed for reproducibility
        same_seed(config['seed'])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        train_loop(config, device)
