import numpy as np
import torch
from torch.autograd import Variable
from data_utils import *  

def testARImage(test_dataset_path, model, info, verbose=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    model.to(device)
    model.eval()
    
    # load test images
    test_image_names = listdir( test_dataset_path )
    num_test_images = len( test_image_names )
    print('Found %d total test images\n'%(num_test_images))
    resx = info['image_size_x']
    resy = info['image_size_y']

    # define the corners of a region on the test images to be destroyed with noise
    destroy_part_x1 = resx // 2
    destroy_part_x2 = resx // 2 + resx // 10
    destroy_part_y1 = resy // 2 - resy // 10
    destroy_part_y2 = resy // 2 + resy // 10
    rec_error = 0. # will store reconstruction error

    for s in range(num_test_images):
        image_full_filename = os.path.join(test_dataset_path, test_image_names[s])
        print('Loading TEST image %d/%d: %s'%(s+1, num_test_images, image_full_filename))                
        im = grayscale_img_load(image_full_filename) / 255.
        im = np.array(im.astype('float32'))
        im = torch.from_numpy(im)
        im = im.to(device)
        
        # load the initial image
        rec_im = torch.zeros(1, 1, info['image_size_x'], info['image_size_y'])
        rec_im[0, :, :, :] = im[0, :, :]
        # destroy the image region with noise
        rand_im = torch.rand(info['image_size_x'], info['image_size_y'])
        rec_im[0, 0, destroy_part_x1:destroy_part_x2, destroy_part_y1:destroy_part_y2] = rand_im[destroy_part_x1:destroy_part_x2, destroy_part_y1:destroy_part_y2]
        rec_im = rec_im.to(device)

        # WRITE CODE HERE TO FIX THE DESTROYED IMAGE REGION
        # USING AN AUTOREGRESSIVE APPROACH
        with torch.no_grad():
            for y in range(destroy_part_y1, destroy_part_y2):
                for x in range(destroy_part_x1, destroy_part_x2):
                    rec_im[0, 0, x, y] = model(rec_im)[0, 0, x, y]

        # measure the reconstruction error        
        diff_im = rec_im[0, 0, destroy_part_x1:destroy_part_x2, destroy_part_y1:destroy_part_y2] - im[0,  destroy_part_x1:destroy_part_x2, destroy_part_y1:destroy_part_y2]
        rec_error_s = np.mean( np.abs(diff_im.cpu().detach().numpy()) )
        print('Rec error %s: %f \n'%(image_full_filename, rec_error_s))
        rec_error += rec_error_s

        # save the reconstructed image
        rec_im = rec_im.squeeze()
        rec_im = rec_im.cpu().detach().numpy()
        final_img = Image.fromarray(np.uint8(rec_im *255.))
        final_img.save(f'{test_image_names[s][:-4]}-rec.jpg', quality=100)
        
    avg_rec_error = rec_error / num_test_images
    print('Avg rec error %f \n'%(avg_rec_error))