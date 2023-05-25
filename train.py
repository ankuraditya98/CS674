import random
import numpy as np
import torch.utils.data as utils_data
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model import PixelCNN
from data_utils import *
from tqdm import tqdm

from torch_geometric.io import read_obj
from torch_geometric.data import Data, DataLoader
import torchaudio as ta
from os import walk
from os.path import join
import pickle

def gumbel_softmax(logprobs, tau=1.0, argmax=False):
    # Decides whether to re-scale probabilities or not based on parameters
    if argmax:
        logits = logprobs/tau
    else:
        g = -torch.log(-torch.log(torch.clamp(torch.rand(logprobs.shape, device=logprobs.device), min=1e-10, max=1)))
        logits = (g + logprobs)/tau
    
    # Gumbel softmax calculation
    soft_labels = torch.softmax(logits, dim=-1)
    labels = soft_labels.detach().argmax(dim=-1, keepdim=True)
    hard_labels = torch.zeros(logits.shape, device=logits.device)
    hard_labels = hard_labels.scatter(-1, labels, 1.0)
    
    # One-hot encoding of the probabilities
    one_hot = hard_labels.detach() - soft_labels.detach() + soft_labels
    
    # Returns one-hot encoding and given labels
    return one_hot, labels.squeeze(-1)


def quantize(logprobs, argmax=False):
    # Calculates the Gumbel softmax of given proabilities
    one_hot, labels = gumbel_softmax(logprobs, argmax=argmax)
    return {"one_hot": one_hot, "labels": labels}

def random_shift(size):
    # Generates a random shift of the input
    return (torch.arange(size) + random.randint(1, size-1))%size

def recon_loss(recon, geom):
    # Simple L2 loss calculation between geometry and reconstruction
    loss = nn.MSELoss()
    return loss(recon, geom)

def landmark_loss(recon, geom, landmarks):
    # Initializes variables
    B, T = geom.shape[0], geom.shape[1]
    
    # Calculates weights for each point
    weights = landmarks[None, None, :, None]
    total = B * T * torch.sum(landmarks) * 3
    
    # Returns the total loss with respect to landmarks
    return torch.sum(((recon - geom) ** 2) * weights)/total

def modality_crossing_loss(audio_cons_recon, exp_cons_recon, geom, mouth_mask, eye_mask):
    # Initializes variables
    B, T = geom.shape[0], geom.shape[1]
    
    # Keeps audio, switches expression
    
    # Calculates weights for each point
    weights = mouth_mask[None, None, :, None]
    total = B * T * torch.sum(mouth_mask) * 3
    
    # Audio consistency loss calculation
    audio_consistency_loss = torch.sum(((audio_cons_recon - geom) ** 2) * weights)/total
    
    # Keeps expression, switches audio
    
    # Calculates weights for each point
    weights = eye_mask[None, None, :, None]
    total = B * T * torch.sum(eye_mask) * 3
    
    # Expression consistency loss calculation
    expression_consistency_loss = torch.sum(((exp_cons_recon - geom) ** 2) * weights)/total
    
    # Returns the sum of the two losses
    return audio_consistency_loss + expression_consistency_loss

def autoregressive_loss(logprobs, target_labels):
    # Simple NLL loss calculation between log of probabilities and targets
    loss = nn.NLLLoss()
    return loss(logprobs.view(-1, logprobs.shape[-1]), target_labels.view(-1))

def load_audio(wave_file: str):
    audio, sr = ta.load(wave_file)
    if not sr == 16000:
        audio = ta.transforms.Resample(sr, 16000)(audio)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    # normalize such that energy matches average energy of audio used in training
    audio = 0.01 * audio / torch.mean(torch.abs(audio))
    return audio

def load_dataset(directory):
    geom_files = []
    audio_files = []
    for dir in tqdm(walk(directory)):
        for file in dir[2]:
                if file.endswith('.obj'):
                    # print('join(dir[0], file):', join(dir[0], file))
                    # loaded_file = np.loadtxt(join(dir[0], file)).astype(np.float32)
                    loaded_file = read_obj(join(dir[0], file))
                    # print(loaded_file.shape)
                    # loaded_file = torch.from_numpy(loaded_file)#.cuda()
                    geom_files.append(Data(pos=loaded_file.pos, face=loaded_file.face))
                    # data_list.append(Data(pos=data.pos, face=data.face))
                if file.endswith('.wav'):
                    # loaded_file = np.loadtxt(join(dir[0], file)).astype(np.float32)
                    # loaded_file = torch.from_numpy(loaded_file)#.cuda()
                    # audio_files = np.append(audio_files, loaded_file)
                    audio_files.append(load_audio(join(dir[0], file)))

    return {'geom': geom_files, 'audio': audio_files}

###############################################################################

# Some samples of how to call the loss functions with appropriate input parameters

mouth_mask_file = np.loadtxt("assets/weighted_mouth_mask.txt").astype(np.float32).flatten()
eye_mask_file = np.loadtxt("assets/weighted_eye_mask.txt", dtype=np.float32).flatten()
landmarks_file = np.loadtxt("assets/eye_keypoints.txt", dtype=np.float32).flatten()

# encoder = encoder.cuda()


mouth_mask = torch.from_numpy(mouth_mask_file).type(torch.float32)#.cuda()
eye_mask = torch.from_numpy(eye_mask_file).type(torch.float32)#.cuda()
landmarks = torch.from_numpy(landmarks_file).type(torch.float32)#.cuda()


# train_dataset = DataReader(segment_length=64)
# val_dataset = DataReader(segment_length=64)

# train_data_directory = './custom_dataset/train'
# val_data_directory = './custom_dataset/val'

# train_data = load_dataset(train_data_directory)
# with open('train_data.pickle', 'wb') as f:
#     # use the pickle.dump() function to pickle the object and write it to the file
#     pickle.dump(train_data, f)

# val_data = load_dataset(val_data_directory)
# with open('val_data.pickle', 'wb') as f:
#     # use the pickle.dump() function to pickle the object and write it to the file
#     pickle.dump(val_data, f)

with open('train_data.pickle', 'rb') as f:
    # use the pickle.load() function to load the pickled object from the file
    train_data = pickle.load(f)
with open('val_data.pickle', 'rb') as f:
    # use the pickle.load() function to load the pickled object from the file
    val_data = pickle.load(f)

batch_size = 32
# train_data_geom_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

print("data['geom'].shape: ", len(train_data['geom']))

# template = data["template"].cuda()
# geom = data["geom"].cuda()
# audio = data["audio"].cuda()
# encoding = encoder.encode(geom, audio)
# recon = reconstruct(template, encoding["expression_code"], encoding["audio_code"])

# recon_loss = recon_loss(recon, geom)

# landmark_loss = landmark_loss(recon, geom, landmarks)

# audio_cons_recon = self._reconstruct(template, encoding["expression_code"][self.random_shift(B), :, :], encoding["audio_code"])
# exp_cons_recon = self._reconstruct(template, encoding["expression_code"], encoding["audio_code"][self.random_shift(B), :, :])
# modality_crossing_loss = modality_crossing_loss(audio_cons_recon, exp_cons_recon, geom, mouth_mask, eye_mask)

# # The encoder is a pre-trained model and is expected to be passed with loaded weights

# encoder = encoder.cuda().eval()

# encoding = self.encoder(geom, audio)
# quantized = quantize(encoding["logprobs"], argmax=True)
# one_hot = quantized["one_hot"].contiguous().detach()
# target_labels = quantized["labels"].contiguous().detach()
# logprobs = model(one_hot, encoding["audio_code"])["logprobs"]
# autoencoder_loss = autoregressive_loss(logprobs, target_labels)

###############################################################################

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