# pads the GTA5 and cityscape images/labels to 2048&1056 recursively
# Put in same directory as all test and train directories

import os
import glob
import cv2
from tqdm import tqdm

scriptdir = os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptdir)
images = glob.glob("*/*.png")

for img in tqdm(images):
    newstr = "padded/" + img
    #print(newstr)
    image=cv2.imread(img)
    #width,length=image.shape[0:2]
    #model uses 2048 × 1056
    if "cityscape" in img:
        #2048 × 1024 for cityscape
        padded = cv2.copyMakeBorder(image,16, 16, 0, 0,cv2.BORDER_CONSTANT, None, value = 0)
        #print("city")
        #print(img)
    else:
        #1914 × 1052 for GTA5
        padded = cv2.copyMakeBorder(image, 2, 2, 67, 67,cv2.BORDER_CONSTANT, None, value = 0)
        #print("GTA5")
        #print(img)
    cv2.imwrite(newstr, padded)
