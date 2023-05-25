import random
import numpy as np
import torch.utils.data as utils_data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from data_utils import *
from tqdm import tqdm

from torch_geometric.io import read_obj
from torch_geometric.data import Data, DataLoader
from decoder import VertexUNet
from encoders import AudioEncoder, ExpressionEncoder, FusionEncoder
from dataset import DataReader
from context_model import ContextModel
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

def reconstruct(encoder, template, expression_code, audio_code, decoder):
        logprobs = encoder(expression_code, audio_code)
        z = quantize(logprobs)["one_hot"]
        recon = decoder(template.unsqueeze(1).expand(-1, z.shape[1], -1, -1).contiguous(), z)
        return recon

###############################################################################

# Some samples of how to call the loss functions with appropriate input parameters

mouth_mask_file = np.loadtxt("assets/weighted_mouth_mask.txt").astype(np.float32).flatten()
eye_mask_file = np.loadtxt("assets/weighted_eye_mask.txt", dtype=np.float32).flatten()
landmarks_file = np.loadtxt("assets/eye_keypoints.txt", dtype=np.float32).flatten()

# encoder = encoder.cuda()

mean = torch.from_numpy(np.load("assets/face_mean.npy"))
stddev = torch.from_numpy(np.load("assets/face_std.npy"))
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

# batch_size = 32
# train_data_geom_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# print("data['geom'].shape: ", len(train_data['geom']))

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

def train(train_data, val_data):
    """
    this function trains an auto-regressive model for image synthesis

    train_dataset_path is the name of the folder that contains images of
    rendered 3D shapes of the training set. 
    val_dataset_path is the name of the folder that contains images of
    rendered 3D shapes of the validation set. 
    
    the function should return a trained convnet
    """
    
    num_epochs = 100
    batch_size = 8
    learning_rate = 1e-4
    weight_decay = 1e-3
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_data = DataReader(mode='train')
    val_data = DataReader(mode='val')
    
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers = 0)
    val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers = 0)
    
    
    expression_encoder = ExpressionEncoder().to(device)
    expression_optimizer = optim.Adam(expression_encoder.parameters(),
                            lr=learning_rate, weight_decay=weight_decay)
    
    audio_encoder = AudioEncoder().to(device)
    audio_optimizer = optim.Adam(audio_encoder.parameters(),
                            lr=learning_rate, weight_decay=weight_decay)
    
    fusion_encoder = FusionEncoder().to(device)
    fusion_optimizer = optim.Adam(fusion_encoder.parameters(),
                            lr=learning_rate, weight_decay=weight_decay)
    
    decoder = VertexUNet(mean=mean, stddev=stddev).to(device)
    decoder_optimizer = optim.Adam(decoder.parameters(),
                            lr=learning_rate, weight_decay=weight_decay)
    
    # print("training_samples: ", training_samples.shape, training_samples)

    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss  = 0
        val_loss = 0
        expression_encoder.train()
        audio_encoder.train()
        fusion_encoder.train()
        decoder.train()
        for i, batch in enumerate(train_data_loader):
            expression_optimizer.zero_grad()
            audio_optimizer.zero_grad()
            fusion_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            B, T = batch['geom'].shape[0], batch['geom'].shape[1]
            
            expression_code = expression_encoder(batch['geom'])
            audio_code = audio_encoder(batch['audio'])
            template = batch['template']
            
            recon = reconstruct(fusion_encoder, template, expression_code, audio_code, decoder)
            
            rec_loss = recon_loss(recon, batch['geom'])

            lmk_loss = landmark_loss(recon, batch['geom'], landmarks)
            
            audio_cons_recon = reconstruct(fusion_encoder, template, expression_code[random_shift(B), :, :], audio_code, decoder)
            exp_cons_recon = reconstruct(fusion_encoder, template, expression_code, audio_code[random_shift(B), :, :], decoder)
            mc_loss = modality_crossing_loss(audio_cons_recon, exp_cons_recon, batch['geom'], mouth_mask, eye_mask)

            # loss calculation
            loss = torch.sum(torch.stack([rec_loss, lmk_loss, mc_loss]))

            # backwards pass
            loss.backward()

            expression_optimizer.step()
            audio_optimizer.step()
            fusion_optimizer.step()
            decoder_optimizer.step()

            train_loss += loss.detach().numpy()

            print('Epoch [%d/%d], Iter [%d/%d], Training loss: %.4f' %(epoch+1, num_epochs, i+1, len(train_data_loader), train_loss/(i+1)))

        train_loss /= i
        
        expression_encoder.eval()
        audio_encoder.eval()
        fusion_encoder.eval()
        decoder.eval()

        for i, batch in enumerate(val_data_loader):
            B, T = batch['geom'].shape[0], batch['geom'].shape[1]
            
            expression_code = expression_encoder(batch['geom'])
            audio_code = audio_encoder(batch['audio'])
            template = batch['template']
            
            recon = reconstruct(fusion_encoder, template, expression_code, audio_code, decoder)
            
            rec_loss = recon_loss(recon, batch['geom'])

            lmk_loss = landmark_loss(recon, batch['geom'], landmarks)
            
            audio_cons_recon = reconstruct(fusion_encoder, template, expression_code[random_shift(B), :, :], audio_code, decoder)
            exp_cons_recon = reconstruct(fusion_encoder, template, expression_code, audio_code[random_shift(B), :, :], decoder)
            mc_loss = modality_crossing_loss(audio_cons_recon, exp_cons_recon, batch['geom'], mouth_mask, eye_mask)

            # loss calculation
            loss = torch.sum(torch.stack([rec_loss, lmk_loss, mc_loss]))

            val_loss += loss.detach().numpy()
            
        val_loss /= i
    
        # report scores per epoch
        print('Epoch [%d/%d], Training loss: %.4f, Validation loss: %.4f'%(epoch+1, num_epochs, train_loss, val_loss))
        
    context_model = ContextModel().to(device)
    context_model_optimizer = optim.Adam(context_model.parameters(),
                            lr=learning_rate, weight_decay=weight_decay)
    
    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss  = 0
        val_loss = 0
        context_model.train()
        for i, batch in enumerate(train_data_loader):
            context_model_optimizer.zero_grad()
            
            B, T = batch['geom'].shape[0], batch['geom'].shape[1]
            
            expression_code = expression_encoder(batch['geom'])
            audio_code = audio_encoder(batch['audio'])
            template = batch['template']
            
            encoding = fusion_encoder(expression_code, audio_code)
            quantized = quantize(encoding, argmax=True)
            one_hot = quantized["one_hot"].contiguous().detach()
            target_labels = quantized["labels"].contiguous().detach()
            logprobs = context_model(one_hot,audio_code)["logprobs"]
            autoencoder_loss = autoregressive_loss(logprobs, target_labels)

            # loss calculation
            loss = torch.sum(autoencoder_loss)

            # backwards pass
            loss.backward()

            context_model_optimizer.step()

            train_loss += loss.detach().numpy()

            print('Epoch [%d/%d], Iter [%d/%d], Training loss: %.4f' %(epoch+1, num_epochs, i+1, len(train_data_loader), train_loss/(i+1)))

        train_loss /= i
        
        context_model.eval()

        for i, batch in enumerate(val_data_loader):
            B, T = batch['geom'].shape[0], batch['geom'].shape[1]
            
            expression_code = expression_encoder(batch['geom'])
            audio_code = audio_encoder(batch['audio'])
            template = batch['template']
            
            encoding = fusion_encoder(expression_code, audio_code)
            quantized = quantize(encoding, argmax=True)
            one_hot = quantized["one_hot"].contiguous().detach()
            target_labels = quantized["labels"].contiguous().detach()
            logprobs = context_model(one_hot,audio_code)["logprobs"]
            autoencoder_loss = autoregressive_loss(logprobs, target_labels)

            # loss calculation
            loss = torch.sum(autoencoder_loss)

            val_loss += loss.detach().numpy()
            
        val_loss /= i
    
        # report scores per epoch
        print('Epoch [%d/%d], Training loss: %.4f, Validation loss: %.4f'%(epoch+1, num_epochs, train_loss, val_loss))