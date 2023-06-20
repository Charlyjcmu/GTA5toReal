import cv2
import glob
from tqdm import tqdm
import numpy as np

out_path = "prepped_frankfurt/"


all_anns = glob.glob("frankfurt_annotations/*gtFine_labelIds.png")
all_anns = sorted(all_anns)

all_imgs = glob.glob("frankfurt_images/*_leftImg8bit.png")
all_imgs = sorted(all_imgs)


pixLabels = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 2, 9: 0, 10: 0, 11: 3, 12: 4, 13: 5, 14: 0, 15: 0, 16: 0, 17: 6, 18: 0, 19: 7, 20: 8, 21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16, 29: 0, 30: 0, 31: 17, 32: 18, 33: 19, -1: 0}


for fn , fnn in tqdm( zip(all_anns , all_imgs ) ):
    si = cv2.imread( fn )
    sii = cv2.imread( fnn )
    di = np.zeros( si.shape ).astype('uint8')
    oho = "_".join((fn.split("\\")[-1].split("_")[:3]))

    allIds = np.unique(si  )
    print(allIds)
    for ii in allIds:
        assert ( ii in pixLabels  )
        di[: , : , 0 ] += ((si[ : , : , 2 ] == ii )*(  pixLabels[ii] )).astype('uint8')
        di[: , : , 1 ] += ((si[ : , : , 2 ] == ii )*(  pixLabels[ii] )).astype('uint8')
        di[: , : , 2 ] += ((si[ : , : , 2 ] == ii )*(  pixLabels[ii] )).astype('uint8')
    assert np.max(di[: , : , 0 ] ) < 20

    cv2.imwrite(out_path+"cityscapes-annotations_prepped_test/" + oho +".png" , di )
    cv2.imwrite(out_path+"cityscapes-images_prepped_test/" + oho +".png" , sii )
