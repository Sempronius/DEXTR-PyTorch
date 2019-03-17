import os

os.chdir('/home/ryan/Documents/DEXTR-PyTorch')

import torch
from collections import OrderedDict
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg') # QTAGG, QT4AGG, Qt5Agg

import nibabel as nib

from torch.nn.functional import upsample

import networks.deeplab_resnet as resnet
from mypath import Path
from dataloaders import helpers as helpers

modelName = 'dextr_pascal-sbd'
pad = 50
thres = 0.8
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

#  Create the network and load the weights
net = resnet.resnet101(1, nInputChannels=4, classifier='psp')
print("Initializing weights from: {}".format(os.path.join(Path.models_dir(), modelName + '.pth')))
state_dict_checkpoint = torch.load(os.path.join(Path.models_dir(), modelName + '.pth'),
                                   map_location=lambda storage, loc: storage)
# Remove the prefix .module from the model when it is trained using DataParallel
if 'module.' in list(state_dict_checkpoint.keys())[0]:
    new_state_dict = OrderedDict()
    for k, v in state_dict_checkpoint.items():
        name = k[7:]  # remove `module.` from multi-gpu training
        new_state_dict[name] = v
else:
    new_state_dict = state_dict_checkpoint
net.load_state_dict(new_state_dict)
net.eval()
net.to(device)


####### Open csv

import pandas as pd
dl_path = '/home/ryan/Documents/Deep_Lesion/'
DL_CSV_PATH = os.path.join(dl_path,"DL_info.csv")

# This is a dataframe of the DL_Info.csv
DL_df = pd.read_csv(DL_CSV_PATH)



import glob
files = glob.glob(dl_path + '/**/*.png', recursive=True)
files_key = []

## make a loop to look at each file
# need to look at each file, check if it is in DL_DF. If not, delete it. 
print('Searching files for key images')
for file in files:
    foldername = os.path.basename(os.path.dirname(file))
    filename = os.path.basename(file)
    comb = foldername + '_' + filename
    x=DL_df.loc[DL_df['File_name'] == comb]  # search DL_DF for current file in files list
    if x.empty == False: #if there is a file name in the dataframe that matches the file (because it is a key image)
        files_key.append(file)
        print('Found Key Image:')
        print(comb)

        
import cv2
###################### In the end will need loop here through all files_key 

###################### THIS IS JUST TEMPORARY TO RANDOMLY SHOW SOME EXAMPLES OF BOUNDED BOXES.
import random
random_num = random.sample(range(1, 100), 10)
###################### This is to just seem some examples and make sure everything looks good
###################### Eventually, this will just be ALL files in files_key. 


for n in random_num:
    print('Example:')
    print(n)
    im = cv2.imread(files_key[n], -1) # -1 is needed for 16-bit image
    foldername = os.path.basename(os.path.dirname(files_key[n]))
    filename = os.path.basename(files_key[n])
    comb = foldername + '_' + filename
    x=DL_df.loc[DL_df['File_name'] == comb]
    print('Filename')
    print(filename)
    # win = list of the two window levels. for example [-175,250]. See above, it is taken directly from DL_df
    def windowing(im, win):
        # scale intensity from win[0]~win[1] to float numbers in 0~255
        im1 = im.astype(float)
        im1 -= win[0]
        im1 /= win[1] - win[0]
        im1[im1 > 1] = 1
        im1[im1 < 0] = 0
        im1 *= 255
        return im1

    import matplotlib.patches as patches



    if x.shape[0] >= 2:
        # create a for loop
        for num_row in range(x.shape[0]):
            ## RELOAD THE IMAGE FILE, there seems to be some issue if we don't do this. End up with a black / blank image. 
            im = cv2.imread(files_key[n], -1) # -1 is needed for 16-bit image
            #############################################
    
            print(filename)
            print('has MULTIPLE bounded boxes')
            
            ###### Get windowing information ################### 
            #win = x.iloc[0][14] another way to do it.
            win = x['DICOM_windows'].iloc[num_row]
            win = win.split(",") # turn it into a list. 
            win = list(map(int, win)) # turn the list of str, into a list of int
            ####################################################
           
            #################################################### APPLYING WINDOWING. Muy importante, otherwise image can look terrible, like just a black box. 
            # there is an offset in the 16-bit png files, intensity - 32768 = Hounsfield unit
            im = im.astype(np.float32, copy=False)-32768  
            im = windowing(im, win).astype(np.uint8)  # soft tissue window
            # This will generate a nice image with appropriate windowing.         
            
            ############################ GET 4 point data
            measurement = x['Measurement_coordinates'].iloc[num_row]
            measurement = measurement.split(",") # turn it into a list. 
            measurement = list(map(float, measurement)) # turn the list of str, into a list of int
            m1_x = round(measurement[0])
            m1_y = round(measurement[1])
            m2_x = round(measurement[2])
            m2_y = round(measurement[3])
            m3_x = round(measurement[4])
            m3_y = round(measurement[5])
            m4_x = round(measurement[6])
            m4_y = round(measurement[7])
            
            ############################
            
            
            ############################ GET BOUNDED BOXES
            bbox = x['Bounding_boxes'].iloc[num_row]
            bbox = bbox.split(",") # turn it into a list. 
            bbox = list(map(float, bbox)) # turn the list of str, into a list of int
            
            #bbox data comes to us as two x,y coordinates... in 4 comma separated values.

            x1 = round(bbox[0])
            y1 = round(bbox[1])
            x2 = round(bbox[2])
            y2= round(bbox[3])
        
            color = np.uint8(np.random.uniform(255, 255, 4)) #This is white. 
            c = tuple(map(int, color))
            #How to make a rectangle 
            Image.fromarray(im) #image without rectangle
            im1 = cv2.rectangle(im,(x1,y1),(x2,y2),color=c)
            ###### ADD IN MEASUREMENTS
            im2 = cv2.circle(im1,(m1_x,m1_y), 2, (255,255,4), -1)
            im3 = cv2.circle(im2,(m2_x,m2_y), 2, (255,255,4), -1)
            im4 = cv2.circle(im3,(m3_x,m3_y), 2, (255,255,4), -1)
            im5 = cv2.circle(im4,(m4_x,m4_y), 2, (255,255,4), -1)
            
            final_image = Image.fromarray(im1) # image with rectangle
            final_image.show()
            
    if x.shape[0] == 1:
        print('Only one bounded box for')
        print(filename)
        
        
        ###### Get windowing information ################### 
        #win = x.iloc[0][14] another way to do it.
        win = x['DICOM_windows'].iloc[0]
        win = win.split(",") # turn it into a list. 
        win = list(map(int, win)) # turn the list of str, into a list of int
        print('Windowing')
        print(win)
        ####################################################
        
        #################################################### APPLYING WINDOWING. Muy importante, otherwise image can look terrible, like just a black box. 
        # there is an offset in the 16-bit png files, intensity - 32768 = Hounsfield unit
        im = im.astype(np.float32, copy=False)-32768  
        im = windowing(im, win).astype(np.uint8)  # soft tissue window
        # This will generate a nice image with appropriate windowing.         
        
        ############################ GET 4 point data
        measurement = x['Measurement_coordinates'].iloc[0]
        measurement = measurement.split(",") # turn it into a list. 
        measurement = list(map(float, measurement)) # turn the list of str, into a list of int
        m1_x = round(measurement[0])
        m1_y = round(measurement[1])
        m2_x = round(measurement[2])
        m2_y = round(measurement[3])
        m3_x = round(measurement[4])
        m3_y = round(measurement[5])
        m4_x = round(measurement[6])
        m4_y = round(measurement[7])
        ############################
        
        
        ############################ GET BOUNDED BOXES
        bbox = x['Bounding_boxes'].iloc[0]
        bbox = bbox.split(",") # turn it into a list. 
        bbox = list(map(float, bbox)) # turn the list of str, into a list of int
        
        #bbox data comes to us as two x,y coordinates... in 4 comma separated values.

        x1 = round(bbox[0])
        y1 = round(bbox[1])
        x2 = round(bbox[2])
        y2= round(bbox[3])
    
        color = np.uint8(np.random.uniform(255, 255, 4)) #This is white. 
        c = tuple(map(int, color))
        #How to make a rectangle 
        print('Image.fromarray(im)')
        Image.fromarray(im) #image without rectangle
        print(Image.fromarray(im))
        im1 = cv2.rectangle(im,(x1,y1),(x2,y2),color=c)
        
        ###### ADD IN MEASUREMENTS
        im2 = cv2.circle(im1,(m1_x,m1_y), 2, (255,255,4), -1)
        im3 = cv2.circle(im2,(m2_x,m2_y), 2, (255,255,4), -1)
        im4 = cv2.circle(im3,(m3_x,m3_y), 2, (255,255,4), -1)
        im5 = cv2.circle(im4,(m4_x,m4_y), 2, (255,255,4), -1)
        
        final_image = Image.fromarray(im1) # image with rectangle
        final_image.show()
               
            
        ######################################## NEXT STEPS
        
        ######################################################   LOAD DEXTR-PYTORCH
        
        ########################################################################## use measurements to automatically  create a mask.
        
        



image = np.array(Image.open('ims/chest-ct-lungs.jpg'))
plt.ion()
plt.axis('off')
plt.imshow(image)
plt.title('Click the four extreme points of the objects\nHit enter when done (do not close the window)')

results = []

with torch.no_grad():
    while 1:
        extreme_points_ori = np.array(plt.ginput(4, timeout=0)).astype(np.int)

        #  Crop image to the bounding box from the extreme points and resize
        bbox = helpers.get_bbox(image, points=extreme_points_ori, pad=pad, zero_pad=True)
        crop_image = helpers.crop_from_bbox(image, bbox, zero_pad=True)
        resize_image = helpers.fixed_resize(crop_image, (512, 512)).astype(np.float32)

        #  Generate extreme point heat map normalized to image values
        extreme_points = extreme_points_ori - [np.min(extreme_points_ori[:, 0]), np.min(extreme_points_ori[:, 1])] + [pad,
                                                                                                                      pad]
        extreme_points = (512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(np.int)
        extreme_heatmap = helpers.make_gt(resize_image, extreme_points, sigma=10)
        extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)

        #  Concatenate inputs and convert to tensor
        input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)
        inputs = torch.from_numpy(input_dextr.transpose((2, 0, 1))[np.newaxis, ...])

        # Run a forward pass
        inputs = inputs.to(device)
        outputs = net.forward(inputs)
        outputs = upsample(outputs, size=(512, 512), mode='bilinear', align_corners=True)
        outputs = outputs.to(torch.device('cpu'))

        pred = np.transpose(outputs.data.numpy()[0, ...], (1, 2, 0))
        pred = 1 / (1 + np.exp(-pred))
        pred = np.squeeze(pred)
        result = helpers.crop2fullmask(pred, bbox, im_size=image.shape[:2], zero_pad=True, relax=pad) > thres

        results.append(result)

        # Plot the results
        plt.imshow(helpers.overlay_masks(image / 255, results))
        plt.plot(extreme_points_ori[:, 0], extreme_points_ori[:, 1], 'gx')
