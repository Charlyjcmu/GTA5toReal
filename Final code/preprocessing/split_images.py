import random
import os


# os.makedirs("images/test")
# os.makedirs("images/train")
# os.makedirs("labels/test")
# os.makedirs("labels/train")

names = ["0000" + str(i) for i in range(1, 2500)]
names = [name[-5:] for name in names]

test_names = random.sample(names, 500)

for name in names:
    if name in test_names:
        os.rename("01_images/images/"+name+".png", "01_images/test_images/"+name+".png")
        os.rename("01_labels/labels/"+name+".png", "01_labels/test_labels/"+name+".png")
    else:
        os.rename("01_images/images/"+name+".png", "01_images/train_images/"+name+".png")
        os.rename("01_labels/labels/"+name+".png", "01_labels/train_labels/"+name+".png")
        