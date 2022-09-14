import os
import csv
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

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

class myDataset(Dataset):
    def __init__(self, test_csv, test_dir, transform):
        data = pd.read_csv(test_csv).values
        
        self.images = data[:,0]
        
        self.target = [6]*data.shape[0] if type(data[0][1]) != int else data[:,1]
        self.data_dir = test_dir
        self.transform = transform

    def __getitem__(self, index):
        fn = self.images[index]
        img = os.path.join(self.data_dir, fn)
        img_pil =  Image.open(img).convert('RGB')
        fn = self.transform(img_pil)
        target = self.target[index]
        return fn, target

    def __len__(self):
        return len(self.images)

def test(test_loader, model, device):
    model.eval()
    preds = []
    for x, _ in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            output = model(x)
            pred = torch.max(output, 1)[1]
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds

def save_csv(preds,config):
    img = pd.read_csv(config['csv_path']).values[:,0]

    with open(config['save_path'], 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['ID', 'Label'])
        
        for x, y in zip(img,preds):
            writer.writerow([x, y])
    print('saved')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_models", help="show the models which can be trained", action="store_true")
    parser.add_argument("-s", "--seed", help="set a seed", type=int, default=1314520)
    parser.add_argument("-b", "--batch_size", help="set the batch size", type=int, default=128)
    parser.add_argument("-cp", "--csv_path", help="the path of test.csv", type=str, default=r'./aoi_data/test.csv')
    parser.add_argument("-sp", "--save_path", help="the path to save predict result", type=str, default=r'./predict_result.csv')
    parser.add_argument("-td", "--test_dir", help="the path test set folder", type=str, default=r'./aoi_data/test_images')
    parser.add_argument("-m", "--model_name", help="choose the model to train", type=str)
    parser.add_argument("-wp", "--weight_path", help="the path of model_weight", type=str)
    args = parser.parse_args()
    
    if args.show_models:
        print(['efficientnet_b6','efficientnet_b7','efficientnet_v2_l','efficientnet_v2_m','efficientnet_v2_s'])
    else:
        config = {
            'seed': args.seed,
            'batch_size': args.batch_size, 
            'csv_path': args.csv_path,
            'save_path': args.save_path,
            'model_name': args.model_name,
            'weight': args.weight_path
        }
        
        transform = transforms.Compose([
            transforms.Resize(128),
            transforms.GaussianBlur(5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.6951, 0.6951, 0.6951],
                std=[0.1011, 0.1011, 0.1011]
            )
        ])

        # Set seed for reproducibility
        same_seed(config['seed'])

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load test data
        test_data = myDataset(config['csv_path'],test_dir=args.test_dir,transform=transform)
        test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

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
            
        model.load_state_dict(torch.load(config['weight'], map_location=device))

        preds = test(test_loader, model, device)
        save_csv(preds,config)