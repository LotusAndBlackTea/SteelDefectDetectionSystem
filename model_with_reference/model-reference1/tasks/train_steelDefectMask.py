''' 4 class Mask Layers

Output: 4 binary mask layers each corresponding to a class

'''

import torch
from torch.utils.data import DataLoader
import os
from utilities.severstalData_utils import SeverstalSteelData
# from networks.tiramisu import FCDenseNet57
import utilities.lossMetrics_utils as LossMet
import segmentation_models_pytorch as smp

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#----

import csv
def log_to_csv(data, csv_file):
    with open(csv_file, "a") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(data)
    csvFile.close()

#----------------------------------------------------

TRAIN_NAME = "DBCE-LinkSEResnxt101"
if not os.path.exists("logs/"+TRAIN_NAME): os.makedirs("logs/"+TRAIN_NAME)
#----

num_epochs = 10000
batch_size = 4
acc_batch = 8 / batch_size
learning_rate = 1e-5

model = smp.Linknet('se_resnext101_32x4d', classes=4, activation=None).to(device)

## --- Pretrained Loader
pretrained_dict = torch.load("weights/DBCE-LinkSEResnxt101_model.pth")
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
print("Pretrained layers Loaded:", pretrained_dict.keys())
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
## ---


diceCrit = LossMet.DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

## --- Loss
criterionDBCE = LossMet.DiceBCELoss()
criterionFTversky = LossMet.FocalTverskyLoss()
criterionFocal = LossMet.FocalLoss()
def loss_estimator(output, target):

    lossDBCE = criterionDBCE(output, target)
#     lossFTversky = criterionFTversky(output, target)
#     lossFocal = criterionFocal(output, target)

    return lossDBCE
#----


DATASET_PATH='datasets/severstal/'

train_dataset = SeverstalSteelData(csv_file='train.csv',
                                    root_dir= DATASET_PATH,
                                    device = device)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=8)

test_dataset = SeverstalSteelData(csv_file='validate.csv',
                                   root_dir=DATASET_PATH,
                                   device = device)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=8)

#-----

if __name__ =="__main__":

    best_loss = float("inf")
    for epoch in range(num_epochs):
        #--- Train
        acc_loss = 0
        running_loss = []
        for ith, (img, gt) in enumerate(train_dataloader):
            img = img.to(device)
            gt = gt.to(device)

            #--- forward
            output = model(img)

            loss = loss_estimator(output, gt) / acc_batch
            acc_loss += loss

            #--- backward
            loss.backward()
            if ( (ith+1) % acc_batch == 0):
                optimizer.step()
                optimizer.zero_grad()
                print('epoch[{}/{}], Mini Batch-{} loss:{:.4f}'
                    .format(epoch+1, num_epochs, (ith+1)//acc_batch, acc_loss.data))
                running_loss.append(acc_loss.data)
                acc_loss=0
                #break
        log_to_csv(running_loss, "logs/"+TRAIN_NAME+"/trainLoss.csv")

        #--- Validate
        val_loss = 0
        val_accuracy = 0
        for jth, (val_img, val_gt) in enumerate(test_dataloader):
            val_img = val_img.to(device)
            val_gt = val_gt.to(device)
            with torch.no_grad():
                val_output = model(val_img)
                val_loss += loss_estimator(val_output, val_gt)

                val_accuracy += (1 - diceCrit(val_output, val_gt))
            #break
        val_loss = val_loss / len(test_dataloader)
        val_accuracy = val_accuracy / len(test_dataloader)

        print('epoch[{}/{}], [-----TEST------] loss:{:.4f}  Accur:{:.4f}'
              .format(epoch+1, num_epochs, val_loss.data, val_accuracy.data))
        log_to_csv([val_loss.item(), val_accuracy.item()],
                    "logs/"+TRAIN_NAME+"/testLoss.csv")

        #--- save Checkpoint
        if val_loss < best_loss:
            print("***saving best optimal state [Loss:{}] ***".format(val_loss.data))
            best_loss = val_loss
            torch.save(model.state_dict(), "weights/"+TRAIN_NAME+"_model.pth")
            log_to_csv([epoch+1, val_loss.item(), val_accuracy.item()],
                    "logs/"+TRAIN_NAME+"/bestCheckpoint.csv")
