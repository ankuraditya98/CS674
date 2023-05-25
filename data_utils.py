import torch
import os
import pickle as pickle
import numpy as np
import gzip
import time
from PIL import Image
from tqdm import tqdm
import torchaudio as ta
from os import walk
from torch_geometric.data import Data
from torch_geometric.io import read_obj
from os.path import join

def save_checkpoint(model, epoch):
    """save model checkpoint"""
    model_out_path = "model/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("model/"):
        os.makedirs("model/")
    torch.save(state, model_out_path)        
    print("Checkpoint saved to {}".format(model_out_path))

def grayscale_img_load(path):
    """ load a grayscale image """
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = np.array(img.convert('L'))
        return np.expand_dims(img, axis = 0) #specific format for pytorch. Expects channel as first dimension

def load_audio(wave_file: str):
    audio, sr = ta.load(wave_file)
    if not sr == 16000:
        audio = ta.transforms.Resample(sr, 16000)(audio)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    # normalize such that energy matches average energy of audio used in training
    audio = 0.01 * audio / torch.mean(torch.abs(audio))
    return audio

# Loads the raw dataset from obj and wav files
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

# Loads the template faces from the dataset
def load_template_dataset(directory):
    temp_files = []
    for dir in tqdm(walk(directory)):
        for file in dir[2]:
                if file.endswith('.obj'):
                    loaded_file = read_obj(join(dir[0], file))
                    temp_files.append(Data(pos=loaded_file.pos, face=loaded_file.face))

    return temp_files

# Loads data from saved pkl files
def load_data(mode='train'):
    if mode=='train':
        with open('train_data.pickle', 'rb') as f:
            # use the pickle.load() function to load the pickled object from the file
            data = pickle.load(f)
            if(len(data)==0):
                print("Error loading training data!")
                return
    else:
        with open('val_data.pickle', 'rb') as f:
            # use the pickle.load() function to load the pickled object from the file
            data = pickle.load(f)
            if(len(data)==0):
                print("Error loading validation data!")
                return
            
    return data

def listdir(path):
    ''' ignore any hidden fiels while loading files in directory'''
    return [f for f in os.listdir(path) if not f.startswith('.')]

def save_data(train_dataset_path, val_dataset_path, verbose = False):
    """
    Load all images in numpy matrix.

    train_dataset_path: is the name of the folder that contains images of the training set
    val_dataset_path: is the name of the folder that contains images of the validation set
    """
    
    # get stats
    info = {}
    train_image_names = listdir( train_dataset_path )
    val_image_names = listdir( val_dataset_path )
    num_train_images = len( train_image_names )
    num_val_images = len( val_image_names )
    print('Found %d total training images\n'%(num_train_images))
    print('Found %d total validation images\n'%(num_val_images))

    # prepare the training image database
    data = {'train_imgs' : [], 'val_imgs' : [] }
    for i in range(num_train_images):          
        image_full_filename = os.path.join(train_dataset_path, train_image_names[i])

        if verbose:
            print('Loading training image %d/%d: %s \n'%(i+1, num_train_images, image_full_filename))
                
        im = grayscale_img_load(image_full_filename)/255.
        data['train_imgs'].append(im.astype('float32'))

    for i in range(num_val_images):          
        image_full_filename = os.path.join(val_dataset_path, val_image_names[i])

        if verbose:
            print('Loading validation image %d/%d: %s \n'%(i+1, num_val_images, image_full_filename))
                
        im = grayscale_img_load(image_full_filename)/255.
        data['val_imgs'].append(im.astype('float32'))

    data['train_imgs'] = np.array(data['train_imgs'])
    data['val_imgs'] = np.array(data['val_imgs'])
    info['image_size_x'] = im.shape[1]
    info['image_size_y'] = im.shape[2]

    '''save as gzip file for better data compression'''
    if verbose:
        print('Saving training/validation images in imgs.npy.gz...')
    t = time.time()
    for item in data:
        print("=> Saving data "+ item)
        f = gzip.GzipFile(item+'.npy.gz', "w")
        np.save(f, data[item])
        f.close()
    if verbose:
        print('Done')    

    '''save info'''
    if verbose:
        print('Saving data info in info.p...')    
    pickle.dump(info, open( "info.p", "wb" ) )
    if verbose:
        print('Done')    

    print("Time Taken %.3f sec"%(time.time()-t))

    return