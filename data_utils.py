import torch
import os
import pickle as pickle
import numpy as np
import gzip
import time
from PIL import Image

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

def load_data(verbose = False):
    '''load saved gzip files'''
    t = time.time()
    files = [f for f in listdir(os.getcwd()) if ".npy.gz" in f]
    if len(files) != 0:
        if verbose:
            print('=> Loading data info from info.p...')
        info = pickle.load( open( "info.p", "rb" ) )
        if verbose:
            print('Done')

        data = {}        
        for item in files:
            if verbose:
                print('=> Loading ' + item + '...')
            f = gzip.GzipFile(item, "r")
            data[item.replace('.npy.gz','')] = np.load(f)
            if verbose:
                print('Done')
        
        if verbose:
            print("Time Taken %.3f sec"%(time.time()-t))
        
        return data, info
    else:
        print("Couldn't open file. Run save_data!")
        return [], []

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