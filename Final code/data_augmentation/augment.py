# needs to install newest version (0.4.0) 
#   pip install git+https://github.com/aleju/imgaug.git
import imgaug as ia
import imgaug.augmenters as iaa
import os
import glob
import imageio
from tqdm import tqdm
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


ia.seed(1)

# augment 1, flip horizontally + rotate
seq1 = iaa.Sequential([iaa.Fliplr(0.5), 
                       iaa.Affine(rotate=(-10, 10))])

# augment 2, contrast change
seq2 = iaa.Sequential([iaa.LogContrast(gain=(0.6, 1.4))])

# augment 3, all
seq3 = iaa.Sequential([iaa.Fliplr(0.5), 
                       iaa.Affine(rotate=(-10, 10)), 
                       iaa.LogContrast(gain=(0.6, 1.4))])

scriptdir = os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptdir)
out_path = "aug/"
all_anns = glob.glob("annotations_prepped_train/*.png")
all_anns = sorted(all_anns)
all_imgs = glob.glob("images_prepped_train/*.png")
all_imgs = sorted(all_imgs)

for fn , fnn in tqdm(zip(all_anns , all_imgs ) ):
    #print(fn)
    si = imageio.imread(fn)
    sii = imageio.imread(fnn)
    #print(image.shape)
    segmap = SegmentationMapsOnImage(si, shape=sii.shape)
    augImg1, augSeg1 = seq1(image=sii, segmentation_maps=segmap)
    imageio.imwrite(out_path+"aug1/"  + fn , augSeg1.get_arr())
    imageio.imwrite(out_path+ "aug1/" + fnn, augImg1)
    augImg2, augSeg2 = seq2(image=sii, segmentation_maps=segmap)
    imageio.imwrite(out_path+"aug2/"  + fn , augSeg2.get_arr())
    imageio.imwrite(out_path+ "aug2/" + fnn, augImg2)
    augImg3, augSeg3 = seq3(image=sii, segmentation_maps=segmap)
    imageio.imwrite(out_path+"aug3/"  + fn , augSeg3.get_arr())
    imageio.imwrite(out_path+ "aug3/" + fnn, augImg3)