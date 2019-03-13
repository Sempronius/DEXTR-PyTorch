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

DL_df = pd.read_csv(DL_CSV_PATH)


x=DL_df.iloc[0]
#### CREATE LOOP HERE TO GO THROUGH 0 to END. And evaluate each filename. 


# or...
x=DL_df.loc[DL_df['File_name'] == '000001_01_01_109.png']




import glob
files = glob.glob(dl_path + '/**/*.png', recursive=True)

## make a loop to look at each file. 
#*********************** SOME FILES WILL NOT BE KEY IMAGES. Need to make an IF then 


foldername = os.path.basename(os.path.dirname(files[0]))
filename = os.path.basename(files[0])
comb = foldername + '_' + filename

x=DL_df.loc[DL_df['File_name'] == comb]

example_filename = os.path.join(dl_path,image_path,'images_png',directory, '000001_02_01_008-023.nii.gz')
img = nib.load(example_filename)
a = np.array(img.dataobj) #get array of img nifti file. 
a1 = Image.fromarray(a[:,:,9]) # 

#  Read image and click the points

# Need to load a nifti file from DL. Load CSV file. Use Nifti file name to find info in CSV. 
# Use info in CSV to automatically find 4 points. 

# load a nifti file. 

############################################################
image_path = 'Images_png_01' 
# probably need to be a little creative here when it comes time to load all times. 
#############################################################
nifti_dir = 'Images_nifti'
dl_path_current = os.path.join(dl_path,image_path,nifti_dir)

#example_filename = os.path.join(dl_path_current, '000001_01_01_103-115.nii.gz')
example_filename = os.path.join(dl_path_current, '000001_02_01_008-023.nii.gz')
img = nib.load(example_filename)
a = np.array(img.dataobj) #get array of img nifti file. 
a1 = Image.fromarray(a[:,:,9]) #???????? 
plt.imshow(a1,cmap="gray",vmin=-175,vmax=275)
############################################### 
########## Take DICOM WINDOWS tab from Csv, and use min and max for vmin and vmax. Image will display appropriately. 
##################### NOT REALLY SURE THAT VMIN AND VMAX ARE THE WAY TO DO THIS?
plt.savefig('savedImage.png') #can't seem to get it to display, so I just save it. 
################ CAN WE FIGURE OUT HOW TO JUST KEEP THIS FILE/ITEM in memory? instead of having to save it? 
###############
import matplotlib.patches as patches
fig,ax = plt.subplots(1)
ax.imshow(a1)
###### NEED TO LOAD CSV AND AUTO LOAD THIS DATA!!!!!! ################### !!!!
#data comes to us as two x,y coordinates... in 4 comma separated values.
x1 = 229
y1 = 258
x2 = 285
y2= 325
###### NEED TO LOAD CSV AND AUTO LOAD THIS DATA!!!!!! ################### !!!!
#first set is top left corner of bounded box.
# Second set is bottom right corner of bounded box.  
## What we need:
# bottom left coordinate, width, height.
# bottom left coordinate:
#x1,y2 (for bottom and left. So take the x value from first set, and y-value from second set)
# width = 
# x-x abs
width = abs(x1-x2)
# height = abs(y1-y2) 
height = abs(y1-y2)

rect = patches.Rectangle((x1,y1),width,height,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)

fig.savefig('savedImage.png')

circ = patches.Circle((272,320),1,edgecolor='r',facecolor='none')
circ1 = patches.Circle((246,263),1,edgecolor='r',facecolor='none')
circ2 = patches.Circle((234,305),1,edgecolor='r',facecolor='none')
circ3 = patches.Circle((280,288),1,edgecolor='r',facecolor='none')


ax.add_patch(circ)
ax.add_patch(circ1)
ax.add_patch(circ2)
ax.add_patch(circ3)
fig.savefig('savedImage.png')

################### NEXT STEPS: Need to figure out if we can display the Measurement Coordinates, to confirm they are what we want. If so, then try and feed them into the code below instead of clicking. 


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
