import numpy as np
import torch.utils.data as utils_data
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model import PixelCNN
from data_utils import *  

def trainARImage(train_dataset_path, val_dataset_path, verbose=False, data_npy_exists = False):
    """
    this function trains an auto-regressive model for image synthesis

    train_dataset_path is the name of the folder that contains images of
    rendered 3D shapes of the training set. 
    val_dataset_path is the name of the folder that contains images of
    rendered 3D shapes of the validation set. 
    
    the function should return a trained convnet
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Read all images, and save them in numpy matrix.
    if not data_npy_exists:
        save_data(train_dataset_path, val_dataset_path, verbose)

    # # Load saved numpy matrix
    data, info = load_data(verbose)

    # Get train data
    train_imgs = data['train_imgs']
    train_imgs = Variable(torch.from_numpy(train_imgs))

    # Get validation data
    val_imgs = data['val_imgs']
    val_imgs = Variable(torch.from_numpy(val_imgs))
    val_imgs = val_imgs.to(device) # keep them in the cuda device

    if(len(train_imgs)==0):
        print("Error loading training data!")
        return
    if(len(val_imgs)==0):
        print("Error loading validation data!")
        return

    # An interactive plot showing how loss function on the training and validation splits
    fig, axes = plt.subplots(ncols=1, nrows=2)
    axes[0].set_title('Training loss')
    axes[1].set_title('Validation loss')
    plt.tight_layout()
    plt.ion()
    plt.show()
    
    # Model/Learning hyperparameter definitions
    model     = PixelCNN().to(device)
    criterion = nn.L1Loss()
    learningRate = 0.001 
    numEpochs    = 20
    weightDecay  = 0.001
    batch_size   = 64
    optimizer    = torch.optim.AdamW(model.parameters(), lr=learningRate, weight_decay=weightDecay)

    training_samples = utils_data.TensorDataset(train_imgs)
    data_loader      = utils_data.DataLoader(training_samples, batch_size=batch_size, shuffle=True, num_workers = 0)

    # print("training_samples: ", training_samples.shape, training_samples)

    print("Starting training...")
    for epoch in range(numEpochs):
        train_loss  = 0
        val_loss = 0
        model.train()
        for i, batch in enumerate(data_loader):
            
            # WRITE CODE HERE TO IMPLEMENT 
            # THE FORWARD PASS AND BACKPROPAGATION
            # FOR EACH PASS ALONG WITH THE L1 LOSS COMPUTATION
            optimizer.zero_grad()
            
            # forward pass
            prediction = model(batch[0])

            # loss calculation
            loss = criterion(prediction, batch[0])

            # backwards pass
            loss.backward()

            optimizer.step()

            train_loss = loss.detach().numpy()

            if verbose:
                print('Epoch [%d/%d], Iter [%d/%d], Training loss: %.4f' %(epoch+1, numEpochs, i+1, len(train_imgs)//batch_size, train_loss/(i+1)))

        # WRITE CODE HERE TO EVALUATE THE LOSS ON THE VALIDATION DATASET
        model.eval()

        # forward pass
        eval_prediction = model(val_imgs)

        # loss calculation
        eval_loss = criterion(eval_prediction, val_imgs)

        val_loss = eval_loss.detach().numpy()

        print("train_loss: ", type(train_loss), train_loss.shape)
        print("val_loss: ", type(val_loss), val_loss.shape)

        # show the plots
        if epoch != 0:            
            axes[0].plot([int(epoch)-1, int(epoch)], [prevtrain_loss, train_loss/(i+1)], marker='o', color="blue", label="train")
            axes[1].plot([int(epoch)-1, int(epoch)], [prevval_loss, val_loss], marker='o', color="red", label="validation")
            plt.pause(0.0001) # pause required to update the graph

        if epoch==1:
            axes[0].legend(loc='upper right')
            axes[1].legend(loc='upper right')

        prevtrain_loss = train_loss/(i+1)
        prevval_loss = val_loss
    
        # report scores per epoch
        print('Epoch [%d/%d], Training loss: %.4f, Validation loss: %.4f'%(epoch+1, numEpochs, train_loss/(i+1), val_loss))

        # save trained models
        save_checkpoint(model, epoch+1)

        # save loss figures
        plt.savefig("error-plot.png")

    return model, info