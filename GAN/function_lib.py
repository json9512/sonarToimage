import os 
import numpy as np
from PIL import Image 


# Crop the image 
# img - img to crop
# pix - number of pixels to crop on all sides
def cropImage(img, pix):
    width, height = img.size
    left = pix
    right = width - pix
    bottom = height - pix
    top = 5
    cropped = img.crop((left,top,right,bottom))

    return cropped

# Creates the final image from the cropped images in the path. 
#   path - directory where the GAN generated images are
#   pix - Pixel value to crop the images on all sides
#
def GenerateFinalImage(path="result/GAN_data64/full_scan/", pix=7):
    all_path = sorted([x for x in os.listdir(path)], key=len)

    # generate the list to store data
    arr_list = [[] for i in range(485)]

    # Store all the images as np.array according to the file number to the arr_list list
    for f in all_path:
        for i in range(485):
            if int(f.split("_")[1].split(".")[0]) == i:
                arr = np.array(cropImage(Image.open(path+f), pix))
                arr_list[i].append(arr)

    final_img = []

    # iterate through the final list (485)
    for i, imgs in enumerate(arr_list):
        # append the images in the imgs list
        arr = np.vstack((imgs[0], imgs[1], imgs[2], imgs[3], imgs[4], imgs[5], imgs[6],  
                        imgs[7], imgs[8], imgs[9], imgs[10], imgs[11]))
        
        final_img.append(arr)

    # Save the final image 
    final_img = np.hstack(([final_img[i] for i in range(len(final_img))]))
    os.makedirs(path, exist_ok=True)
    Image.fromarray(np.uint8(final_img)).save("final_image.png")
    print("\n[INFO] Final Image saved")
