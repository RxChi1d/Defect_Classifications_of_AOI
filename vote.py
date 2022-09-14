import os
import csv
import argparse
import numpy as np
import pandas as pd


class predict_dict():
    def __init__(self,csv_folder):
        # 將model的predict寫成dict
        csv_files = listdir_nohidden(csv_folder)
        model_predict = {}
        for i in range(len(csv_files)):
            csv_file = os.path.join(csv_folder,csv_files[i])
            if i == 0:
                self.imgs = (pd.read_csv(csv_file).values[:,0])
            label = (pd.read_csv(csv_file).values[:,1])
            model_predict[os.path.splitext(csv_file)[0]] = label
        self.model_predict = model_predict
        
    def __getTestSetSize__(self):
        # 回傳測試集size
        for i in self.model_predict.values():
            return (len(i))
    def __getImgsList__(self):
        return self.imgs
    
    def __getModelList__(self):
        # 讀取所有的model names
        return self.model_predict.keys()
    
    def __getpredict__(self,modelName):
        # 回傳指定model的predict
        return self.model_predict[modelName]

def listdir_nohidden(path):
    files = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            files.append(f)
    return sorted(files)

def vote(csv_folder):
    predicts = predict_dict(csv_folder)

    # 創建一個array為[img,labels]
    amount = np.zeros((predicts.__getTestSetSize__(),6),int)

    for model in predicts.__getModelList__():
        # 取出每個model的predict
        predict = predicts.__getpredict__(model)
        # 投票，將對應的label值加一
        for idx in range(len(predict)):
            amount[idx][predict[idx]] += 1

    # 將票數最多者定為該img的預測結果
    result = []
    for a in amount:
        m = np.max(a)
        idx = np.where(a==m)[0][0]
        result.append(idx)

    # 儲存最終結果
    img = predicts.__getImgsList__()

    with open('final.csv', 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['ID', 'Label'])
        
        for x, y in zip(img,result):
            writer.writerow([x, y])
    print('saved')  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--csv_folder", help="the folder of test csvs", type=str)
    args = parser.parse_args()
    
    vote(args.csv_folder)
