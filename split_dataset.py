import os
import argparse
from shutil import copyfile
from os.path import join

def listdir_nohidden(path):
    files = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            files.append(f)
    return sorted(files)


def split_data(root, save_path, valid_ratio):
    contents = sorted(listdir_nohidden(root))
    # create train and valid folder
    for target in ['train_data','valid_data']:
        folder = join(save_path,target)
        if not os.path.isdir(folder):
            os.mkdir(folder)
        for i in range(len(contents)):
            label_folder = join(folder,str(i))
            if not os.path.isdir(label_folder):
                os.mkdir(label_folder)
    
    for content in contents:
        origin_folder = join(root,content)
        files = listdir_nohidden(origin_folder)
        length = len(files)
        train_len = int(length*(1-valid_ratio))
        
        # split train set and copy to new folder
        for idx in range(train_len):
            origin_img = join(origin_folder,files[idx])
            img_path = join(join(join(save_path,'train_data'),content),files[idx])
            copyfile(origin_img,img_path)
        
        # split valid set and copy to new folder
        for idx in range(train_len,length):
            origin_img = join(origin_folder,files[idx])
            img_path = join(join(join(save_path,'valid_data'),content),files[idx])
            copyfile(origin_img,img_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_data", help="the folder of all train_data(split by each class)", type=str)
    parser.add_argument("-s", "--save_path", help="the path to save new train data", type=str)
    parser.add_argument("-r", "--valid_ratio", help="set the valid ratio[0~1]", type=float, default=0.2)
    args = parser.parse_args()

    split_data(args.train_data, args.save_path, args.valid_ratio)
    