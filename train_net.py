import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import os
import argparse
from rich.progress import track

from dataset.HDR_data import getTrainVal_loader
from models.model import Salicon

#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,4,7"

def train_model(model, dataloaders, criterion, optimizer, num_epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    for epoch in range(num_epochs):
        for phase in ['train']:
            running_loss=0
            if phase=='train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            total_samples = 0

            for _, (inputs, labels, names) in track(enumerate(dataloaders[phase]), total=len(dataloaders[phase]), description=f'Epoch {epoch+1} {phase}'):
                inputs = [input.to(device) for input in inputs]
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.autograd.set_grad_enabled(phase=='train'):
                    outputs = model(*inputs)
                    loss1 = criterion(outputs,labels)
                    loss_fc_2 = nn.CrossEntropyLoss()
                    loss2 = loss_fc_2(outputs, labels)

                    loss = loss1+loss2
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss+=loss.item()

                #running_loss+=loss.item()*inputs[0].size(0)

            epoch_loss=running_loss/len(dataloaders[phase])

            print("{} Loss: {}".format(phase,epoch_loss))

        output_dir = './checkpoints/'
        os.makedirs(output_dir, exist_ok=True)
        if epoch % 5 == 0:
            save_path  = os.path.join(output_dir, f'salicon_{epoch}.pth')
            torch.save(model.state_dict(),save_path)


def main():
    parser=argparse.ArgumentParser()
    np.random.seed(7777777)

    # train argument
    parser.add_argument('--batch_size', type=int, default=1, help='Input batch size for training')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate for training')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay for SGD')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train')
    parser.add_argument('--checkpoints', type=bool, default=False, help='Number of epochs to train')
    args=parser.parse_args()

    train_dataset_dir='./IMLHDR'
    train_img_dir = 'image_mantiuk'
    train_mutilexp_dir = 'multi_exposure'
    train_label_dir = 'density'
    dataloaders=getTrainVal_loader(train_dataset_dir,train_img_dir,train_mutilexp_dir,train_label_dir)


    model = Salicon()
    # upscale = 4
    # window_size = 8
    # height = (256 // upscale // window_size + 1) * window_size
    # width = (256 // upscale // window_size + 1) * window_size
    # model = Salicon(upscale=2, img_size=(height, width), in_chans=18,
    #               window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
    #               embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2)
  
    model.cuda()
 
    optimizer_ft=optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.decay,nesterov=True)
    criterion_ft=nn.BCELoss()

    if args.checkpoints == True:
        path = r"/data/jiaoshengjie/Code/Saliency_transformer/checkpoints/salicon_150.pth"
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)
        #optimizer_ft.load_state_dict(checkpoint['optimizer'])

    train_model(model, dataloaders, criterion_ft, optimizer_ft, num_epochs=args.epochs)


if __name__ == '__main__':
    main()