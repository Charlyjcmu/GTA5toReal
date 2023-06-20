Inside the "padded" directory is GTA5 and cityscapes padded to same size.
They are 2048x1056 with 20 classes.

Inside the "aug" directory is GTA5 training dataset augmented
"aug1" has Geometric transformations:
    50% flip horizontally
    rotate (+-10) degrees
"aug2" has texture transformations:
    log contrast change (0.6,1.4)
"aug3" has all transformations

There are 1200 training images from GTA5, 
          200 test images from GTA5, 
          and 200 val images from cityscape.

There should be 3 models and 4 datasets, so 24 loss and accuracy graphs??
