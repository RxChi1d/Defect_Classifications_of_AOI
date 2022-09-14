import os
import argparse
import pandas as pd
from shutil import copyfile


def create_folder(save_path):
    # create folders of each label
    for i in range(6):
        path = os.path.join(save_path,str(i))
        if not os.path.isdir(path):
            os.mkdir(path)
                
def split_dataset(train_csv,origin_path,save_path):
    # read imgs and labels
    imgs = pd.read_csv(train_csv).values[:,0]
    labels = pd.read_csv(train_csv).values[:,1]
    for i in range(len(imgs)):
        label = int(labels[i])
        origin_img = os.path.join(origin_path,imgs[i])
        new_img = os.path.join(os.path.join(save_path,str(label)),imgs[i])
        copyfile(origin_img,new_img)
        
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_csv", help="the folder of all train_data(split by every class)", type=str)
    parser.add_argument("-o", "--origin_path", help="set the valid ratio[0~1]", type=str)
    parser.add_argument("-s", "--save_path", help="the path to save new train data and valid data", type=str)
    args = parser.parse_args()
    
    create_folder(args.save_path)
    split_dataset(args.train_csv,args.origin_path,args.save_path)