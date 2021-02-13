#!/usr/bin/env python3
import time
import torch
from models.mps import MPS
from torchvision import transforms, datasets
import pdb
from data.datasets import *
from utils.tools import *
import argparse
from carbontracker.tracker import CarbonTracker
from sklearn.metrics import precision_recall_curve, average_precision_score
from torchvision.utils import save_image
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Globally load device identifier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(loader,optThresh=0.5,testMode=False,plot=False,mode='Valid',post=False):
    ### Evaluation function for validation/testing
    vl_acc = torch.Tensor([0.]).to(device)
    vl_loss = 0.
    labelsNp = [] 
    predsNp = [] 
    model.eval()

    for i, (inputs, labels) in enumerate(loader):
        b = inputs.shape[0]
        labelsNp = labelsNp + labels.numpy().tolist()
        # Make patches on the fly
        inputs = inputs.unfold(2,dim[0],dim[1]).unfold(3,dim[0],\
                dim[1]).reshape(-1,nCh,dim[0],dim[1])
        b = inputs.shape[0]
        # Flatten to 1D vector as input to MPS
        inputs = inputs.view(b,nCh,-1)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Inference
        scores = torch.sigmoid(model(inputs))
        scores[scores.isnan()] = 0
        # Put patches back together
        scores = fold2d(scores.view(-1,dim[0],dim[1]),labels.shape[0])

        preds = scores.clone()

        loss = loss_fun(scores.view(-1,H,W), labels) 
        
        predsNp = predsNp + preds.cpu().numpy().tolist()
        vl_loss += loss.item()
        vl_acc += accuracy(labels,preds.view(-1,H,W))

    # Compute AUC over the full (valid/test) set
    labelsNp, predsNp = np.array(labelsNp), np.array(predsNp)
    fpr, tpr, thresh = precision_recall_curve(labelsNp.reshape(-1), predsNp.reshape(-1))

    if not testMode:
        alpha = 0.7
        gmeans = 2*fpr*tpr/(fpr+tpr)
        gmeans[np.isnan(gmeans)] = 0
        idx = np.argmax(gmeans)
        optThresh = thresh[idx]
        print("Opt Thresh: %.4f with Acc %.4f"%(thresh[idx],gmeans[idx]))
    acc_, acc_sample = accuracy(torch.Tensor(labelsNp),\
            torch.Tensor((predsNp >= optThresh).astype(float)),True)
    acc_std = torch.std(acc_sample)

    if mode is 'Test':    
        acc_sample = acc_sample.cpu().data.numpy()
        print("Min.%.4f [%d]"%(acc_sample.min(),np.argmin(acc_sample)))
        print("Max.%.4f [%d]"%(acc_sample.max(),np.argmax(acc_sample)))
        if args.nuseg:
            idx = [2,8]
            print("2:%.4f,8:%.4f"%(acc_sample[idx[0]],acc_sample[idx[1]]))

    print(mode+" Acc: %.2f +/- %.2f"%(acc_,acc_std))

    vl_acc = average_precision_score(labelsNp.reshape(-1), predsNp.reshape(-1))
    vl_loss = vl_loss/len(loader)
    
    if plot:
            if args.nuseg:
                k = 8
                tmp =  torch.zeros(k,3,H,W).to(device)
                idx = list(np.arange(k))
                preds = torch.Tensor(predsNp[idx]).to(device)
                labels = torch.Tensor(labelsNp[idx]).to(device)
            else:
                k = 32
                k = (labels.shape[0] if labels.shape[0] < k else k)
                tmp =  torch.zeros(k,3,H,W).to(device)
            pred =  ((preds[:k].view(-1,H,W) >= optThresh).float() \
                    + 2*labels[:k])
            ### FN
            tmp[:k,0,:,:][pred==1] = 0.55
            tmp[:k,1,:,:][pred==1] = 0.29
            tmp[:k,2,:,:][pred==1] = 0.39
            ### FP
            tmp[:k,0,:,:][pred==3] = 0.13
            tmp[:k,1,:,:][pred==3] = 0.60
            tmp[:k,2,:,:][pred==3] = 0.20
            ## TP
            tmp[:k,0,:,:][pred==2] = 0.6
            tmp[:k,1,:,:][pred==2] = 0.6
            tmp[:k,2,:,:][pred==2] = 0.6

            save_image(tmp,'vis/ep'+repr(epoch)+'.jpg')
    labelsNp, predsNp = np.array(labelsNp), np.array(predsNp)
    return vl_acc, vl_loss, optThresh


#### MAIN STARTS HERE ####

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--fold', type=int, default=0, help='Fold to use for testing')
parser.add_argument('--feat', type=int, default=4, help='Number of local features')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--l2', type=float, default=0, help='L2 regularisation')
parser.add_argument('--p', type=float, default=0.5, help='Augmentation probability')
parser.add_argument('--aug', action='store_true', default=False, help='Use data augmentation')
parser.add_argument('--save', action='store_true', default=False, help='Save model')
parser.add_argument('--cxr', action='store_true', default=False, help='Use the Lung CXR data.')
parser.add_argument('--nuseg', action='store_true', default=False, help='Use the MO-nuclei seg. dataset')
parser.add_argument('--data_path', type=str, default='lidc/',help='Path to data.')
parser.add_argument('--bond_dim', type=int, default=2, help='MPS Bond dimension')
parser.add_argument('--kernel', type=int, default=4, help='Stride of squeeze kernel')
parser.add_argument('--seed', type=int, default=1, help='Random seed')

# Visualization and log dirs
if not os.path.exists('vis'):
    os.mkdir('vis')
if not os.path.exists('logs'):
    os.mkdir('logs')
logFile = 'logs/'+time.strftime("%Y%m%d_%H_%M")+'.txt'
makeLogFile(logFile)

args = parser.parse_args()

# Assign script args to vars
torch.manual_seed(args.seed)
batch_size = args.batch_size
kernel = args.kernel 
feature_dim = args.feat

### Data processing and loading....
trans_valid = A.Compose([ToTensorV2()])
if args.aug:
    trans_train = A.Compose([A.ShiftScaleRotate(shift_limit=0.5, \
            scale_limit=0.5, rotate_limit=30, p=args.p),ToTensorV2()])
    print("Using Augmentation with p=%.2f"%args.p)
else:
    trans_train = trans_valid
    print("No augmentation....")
   
if args.cxr:
    print("Using Lung CXR dataset")
    print("Using Fold: %d"%args.fold)
    dataset_valid = lungCXR(split='Valid', data_dir=args.data_path, 
                        transform=trans_valid,fold=args.fold)
    dataset_train = lungCXR(split='Train', data_dir=args.data_path,fold=args.fold, 
                                    transform=trans_train)
    dataset_test = lungCXR(split='Test', data_dir=args.data_path,fold=args.fold,
                    transform=trans_valid)
elif args.nuseg:
    print("Using MONuSeg dataset")
    dataset_valid = MoNuSeg(split='Valid', data_dir=args.data_path, 
                        transform=trans_valid)
    dataset_train = MoNuSeg(split='Train', data_dir=args.data_path,
                                        transform=trans_train)
    dataset_test = MoNuSeg(split='Test', data_dir=args.data_path,
                        transform=trans_valid)
else:
    print("Choose a dataset!")
    sys.exit()

# Initiliaze input dimensions
dim = torch.ShortTensor(list(dataset_valid[0][0].shape[1:]))
nCh = int(dataset_valid[0][0].shape[0])
H = dim[0] 
W = dim[1] 
output_dim = H*W # Same as the number of pixels

num_train = len(dataset_train)
num_valid = len(dataset_valid)
num_test = len(dataset_test)
print("Num. train = %d, Num. val = %d, Num. test = %d"%(num_train,num_valid,num_test))

# Initialize dataloaders
loader_train = DataLoader(dataset = dataset_train, drop_last=False,num_workers=1, 
                          batch_size=batch_size, shuffle=True,pin_memory=True)
loader_valid = DataLoader(dataset = dataset_valid, drop_last=True,num_workers=1,
                          batch_size=batch_size, shuffle=False,pin_memory=True)
loader_test = DataLoader(dataset = dataset_test, drop_last=True,num_workers=1,
                         batch_size=batch_size, shuffle=False,pin_memory=True)
nValid = len(loader_valid)
nTrain = len(loader_train)
nTest = len(loader_test)

# Initialize the models
dim = dim//args.kernel 
print("Using Strided Tenet with patches of size",dim)
output_dim = torch.prod(dim)
model = MPS(input_dim=torch.prod(dim), 
        output_dim=output_dim, 
        bond_dim=args.bond_dim,
        feature_dim=feature_dim*nCh,
        lFeat=feature_dim)
model = model.to(device)

# Initialize loss and metrics
accuracy = dice
if args.cxr:
    loss_fun = dice_loss()
else:
    loss_fun = torch.nn.BCELoss(reduction='mean')

# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
                             weight_decay=args.l2)

nParam = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters:%d"%(nParam))
print(f"Maximum MPS bond dimension = {args.bond_dim}")
print(f"Using Adam w/ learning rate = {args.lr:.1e}")
print("Local feature map dim: %d, nCh: %d, B:%d"%(feature_dim,nCh,batch_size))
with open(logFile,"a") as f:
    print("Bond dim: %d"%(args.bond_dim),file=f)
    print("Number of parameters:%d"%(nParam),file=f)
    print(f"Using Adam w/ learning rate = {args.lr:.1e}",file=f)
    print("Local feature map dim: %d, nCh: %d, B:%d"%(feature_dim,nCh,batch_size),file=f)

# Miscellaneous initialization
start_time = time.time()
maxAuc = -1
minLoss = 1e3
convCheck = 10
convIter = 0

# Instantiate Carbontracker
tracker = CarbonTracker(epochs=args.num_epochs,
            log_dir='carbontracker/',monitor_epochs=-1)

# Training starts here
for epoch in range(args.num_epochs):
    tracker.epoch_start()
    running_loss = 0.
    running_acc = 0.
    t = time.time()
    model.train()
    predsNp = [] 
    labelsNp = []
    bNum = 0
    for i, (inputs, labels) in enumerate(loader_train):
        for p in model.parameters():
            p.grad = None
        bNum += 1
        b = inputs.shape[0]
        # Make patches on the fly
        inputs = inputs.unfold(2,dim[0],dim[1]).unfold(3,dim[0],\
                dim[1]).reshape(-1,nCh,dim[0],dim[1])
        labels = labels.unfold(1,dim[0],dim[1]).unfold(2,dim[0],\
                dim[1]).reshape(-1,dim[0],dim[1])
        b = inputs.shape[0]
        # Flatten to 1D vector as input to MPS
        inputs = inputs.view(b,nCh,-1)
        labelsNp = labelsNp + (labels.numpy()).tolist()

        inputs = inputs.to(device)
        labels = labels.to(device)

        scores = torch.sigmoid(model(inputs))

        loss = loss_fun(scores.view(-1), labels.view(-1)) 

        # Backpropagate and update parameters
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = scores.clone()
            predsNp = predsNp + (preds.data.cpu().numpy()).tolist()
            running_acc += accuracy(labels,preds)
            running_loss += loss
            
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, args.num_epochs, i+1, nTrain, loss.item()))
    
    tr_acc = running_acc/nTrain

    # Evaluate on Validation set 
    with torch.no_grad():
        if tr_acc.isnan():
            print('NaN error!')
            break
        vl_acc, vl_loss, optThresh = evaluate(loader_valid,testMode=True,plot=False)
        if vl_acc > maxAuc or vl_loss < minLoss:
            convIter = 0
            if args.save:
                torch.save(model.state_dict(),'saved_models/'+logFile.replace('.txt','.pt'))
            if (vl_acc > maxAuc) or (vl_acc >= maxAuc and vl_loss < minLoss):
                ### Predict on test set if new optimum
                maxAuc = vl_acc
                print('New Best: %.4f'%np.abs(maxAuc))
                ts_acc, ts_loss, _ = evaluate(loader=loader_test,\
                        optThresh=optThresh,testMode=True,plot=True,mode='Test')
                print('Test Set Loss:%.4f\t Acc:%.4f'%(ts_loss, ts_acc))
                with open(logFile,"a") as f:
                    print('Test Set Loss:%.4f\tAcc:%.4f'%(ts_loss, ts_acc),file=f)
                convEpoch = epoch
            elif vl_loss < minLoss:
                minLoss = vl_loss
        else:
            convIter += 1
        if convIter == convCheck:
            print("Converged at epoch:%d with AUC:%.4f"%(convEpoch+1,maxAuc))
            break
    writeLog(logFile, epoch, running_loss/bNum, tr_acc,
            vl_loss, np.abs(vl_acc), time.time()-t)
    tracker.epoch_end()
tracker.stop()
