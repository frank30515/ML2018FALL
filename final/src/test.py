import os 
import time 
import json 
import torch 
import random 
import warnings
import torchvision
import numpy as np 
import pandas as pd 
import imp

from utils import *
from data import HumanDataset
from tqdm import tqdm 
from config import config
from datetime import datetime
from model import *
from torch import nn,optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.metrics import f1_score

# Random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

if not os.path.exists("./logs/"):
    os.mkdir("./logs/")

# Testing
def test(test_loader,model,folds):
    sample_submission_df = pd.read_csv("./sample_submission.csv")

    filenames,labels ,submissions= [],[],[]
    model.cuda()
    model.eval()
    submit_results = []
    for i,(input,filepath) in enumerate(tqdm(test_loader)):
        filepath = [os.path.basename(x) for x in filepath]
        with torch.no_grad():
            image_var = input.cuda(non_blocking=True)
            y_pred = model(image_var)
            label = y_pred.sigmoid().cpu().data.numpy()
           
            labels.append(label > 0.15)
            filenames.append(filepath)

    for row in np.concatenate(labels):
        subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
        submissions.append(subrow)
    sample_submission_df['Predicted'] = submissions
    sample_submission_df.to_csv('./%s_bestloss_submission.csv'%config.model_name, index=None)

# Main function
def main():
    fold = 0
    # Get model
    model = get_net()
    model.cuda()
    # criterion
    optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss().cuda()

    test_files = pd.read_csv("./sample_submission.csv")
    test_gen = HumanDataset(test_files,config.test_data,augument=False,mode="test")
    test_loader = DataLoader(test_gen,1,shuffle=False,pin_memory=True,num_workers=4)

    scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
    start = timer()
    
    best_model = torch.load("./%s_fold_%s_model_best_loss.pth.tar"%(config.model_name,str(fold)))
    model.load_state_dict(best_model["state_dict"])
    test(test_loader,model,fold)
    
if __name__ == "__main__":
    main()