# Environment
- OS: Ubuntu 20.04.2 LTS x86_64  
- Kernel: 5.4.0-122-generic  
- CPU: Intel i7-10700 (16) @ 4.800GHz  
- GPU: MSI GeForce RTX™ 3090 SUPRIM X 24G  
- Memory: 31007MiB  

# Library
You can run  `pip install -r ./requirements.txt` to install the following library  
| Library |
|  ----  |
torch
argparse
torchvision
numpy
Pillow
tqdm
sklearn
torchsummary
torchviz
matplotlib
pandas

# Code
> Tips: our code included the argparse, so if you don't know what kinds of the parameters you can use, please run `python file.py -h` and you will get the detail imformation.

> WARNING: because the default hyperparameters are for the rtx 3090, please adjest them appropriately.

## Split Dataset
The first, because we split the dataset manualy, you need to split it as following:
1. run `python separate_class.py [options]` to separate your train_images by classes
2. run `python split_dataset.py [options]` to split the dataset by the valid ratio

## Train
We trained 8 models in total when the process of expriment, however, because of voting, there are only 5 models which is offered in the model folder, and it get the best perfomance in our expriment.

We adjust the output layer for matching the output with the labels from 1000 to 6.
|Models|
|---|
efficientnet_b6
efficientnet_b7
efficientnet_v2_l
efficientnet_v2_m
efficientnet_v2_s

If you want to train them, you can run `python train.py -m model_name [options]`, and it will save the log and weight of best epoch(lowest valid loss) and the log of tensorboard.

### Training Result
|Model|Epoch|Valid Accuracy|Valid Loss|
|----|----|----|----|
|Resnet50|807|100.0|0.0021078815884578717|
|Regnet_y_16gf|954|100.0|0.00036125763199379435|
|Regnet_y_32gf|85|100.0|0.0005281690953893303|
|efficientnet_v2_l|623|100.0|2.8262531259315438e-05|
|efficientnet_v2_m|419|100.0|0.0006482427881593367|
|efficientnet_v2_s|264|100.0|0.00035412256602285197|
|efficientnet_b7|922|100.0|0.1.0256626202576058e-05|
|efficientnet_b6|694|100.0|2.932406665934195e-06|

## Prediction
To predict the test_images, run `python predict_test.py -m model_name [options]`, and you will get the predict result of the model.
### Test Result
|Model|Test Accuracy|
|----|----|
|Resnet50|0.9950678|
|Regnet_y_16gf|0.9948212|
|Regnet_y_32gf|0.9940813|
|efficientnet_v2_l|0.9955610|
|efficientnet_v2_m|0.9967940|
|efficientnet_v2_s|0.9953144|
|efficientnet_b7|0.9965474|
|efficientnet_b6|0.9958076|

## Voting
Finally, there is a voting as the last step. Please put all of the predict_result.csv in a folder together, and run `python vote.py -d csv_folder`, then you will get the final result.

### Result of Voting
|Model|Test Accuracy|
|----|----|
|8 models|0.9970406|
|5 models|0.9980217|

# Exprimental Result
Our project get the test accuracy as 0.998027(Rank 2) on Sep 5th, 2022.