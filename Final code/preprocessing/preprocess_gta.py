import cv2
import glob
from tqdm import tqdm
import numpy as np


out_path = "prepped/"


all_anns = glob.glob("01_labels/*/*.png")
all_anns = sorted(all_anns)

all_imgs = glob.glob("01_images/*/*.png")
all_imgs = sorted(all_imgs)


# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

pixLabels = {"000":0, "074111":0, "81081":0, "12864128":1, "23235244":2, "160170250": 0, "140150230":0,
"707070":3, "156102102":4, "153153190": 5, "180165180":0, "100100150":0, "90120150":0, "153153153":6, "30170250": 7, "0220220": 8, "35142107":9,
"152251152":10, "18013070":11, "6020220":12, "00255":13, "14200":14, "7000":15, "100600": 16, "9000": 0, "11000": 0, "100800": 17, "23000": 18, "3211119": 19, "202020": 0}


for fn , fnn in tqdm( zip(all_anns , all_imgs ) ):
    si = cv2.imread( fn )
    sii = cv2.imread( fnn )
    di = np.zeros( si.shape ).astype('uint8')
    oho = "_".join((fn.split("/")[-1].split("_")[:3]))
    oho = str(fn[-9:])

    allIds = np.unique(si.reshape(-1, si.shape[2]), axis=0)

    for ii in allIds:

        string_key = "".join(str(x) for x in ii.tolist())
        if (string_key not in pixLabels):
            print(ii)
            print(string_key)
            string_key = "000"

        match0 = (si[ : , : , 0 ] == ii[0] )
        match1 = (si[:, :, 1] == ii[1])
        match2 = (si[:, :, 2] == ii[2])
  
        match = np.logical_and(np.logical_and(match0, match1), match2)

        di[: , : , 0 ] += ((match)*(  pixLabels[string_key] )).astype('uint8')
        di[: , : , 1 ] += ((match)*(  pixLabels[string_key] )).astype('uint8')
        di[: , : , 2 ] += ((match)*(  pixLabels[string_key] )).astype('uint8')


    if "\\train\\" in fn:
        print('writing train' + oho)
        cv2.imwrite(out_path+"annotations_prepped_train/" + oho , di )
        cv2.imwrite(out_path+"images_prepped_train/" + oho , sii )
    elif "\\val\\" in fn:
        cv2.imwrite(out_path+"annotations_prepped_val/" + oho, di )
        cv2.imwrite(out_path+"images_prepped_val/" + oho, sii )
    elif "\\test\\" in fn:
        print('writing test' + oho)
        cv2.imwrite(out_path+"annotations_prepped_test/" + oho, di )
        cv2.imwrite(out_path+"images_prepped_test/" + oho, sii )
    else:
        assert False