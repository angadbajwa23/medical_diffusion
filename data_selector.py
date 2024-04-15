import pandas as pd
import os, shutil
import random
random.seed(42)

df = pd.read_csv('ham/GroundTruth.csv')

num_train_samples = 100
num_val_samples = 15
class_to_img = {}


for column in df.columns[1:]:  # Exclude the 'image' column
    class_images = df[df[column] == 1]['image'].tolist()  # Get the image names for the current class
    random.shuffle(class_images)
    class_to_img[column] = class_images
    # class_to_img[column] = sorted(class_images)


def copy_images(parent_dir, train=True):
    for label, images in class_to_img.items():
        label_dir = os.path.join(parent_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        if train:
            images = images[:num_train_samples]
        else:
            images = images[num_train_samples+1 :num_val_samples]
        for image_name in images:
            source_path = os.path.join('ham/images', image_name+".JPG")  # Assuming images are located in 'data' folder
            destination_path = os.path.join(label_dir, image_name+".jpg")
            shutil.copy(source_path, destination_path)

copy_images("train")
copy_images("val", train=False)



#%%
############## Not Used ##############
## Without class folders directly into train and val

# def copy_data(output_folder, samples):
#     os.makedirs(output_folder, exist_ok=True)
#     for image_name in samples:
#         source_path = os.path.join('ham/images', image_name + '.jpg')  # Assuming image files have extension '.jpg'
#         destination_path = os.path.join(output_folder, image_name + '.jpg')
#         shutil.copy(source_path, destination_path)  # Copy the image to the new folder

# train_samples, val_samples = [], []
# for c in class_to_img:
#     train_samples.extend(class_to_img[c][:num_train_samples])
#     val_samples.extend(class_to_img[c][num_train_samples+1: num_train_samples+num_val_samples])

# # Shuffle the final list
# print(train_samples[1],train_samples[55], train_samples[-1],"\n")

# random.shuffle(train_samples)
# random.shuffle(val_samples)

# print(train_samples[1],train_samples[55], train_samples[-1])
# print(val_samples[1],val_samples[5], val_samples[-1])

# # ISIC_0027719 ISIC_0024994 ISIC_0031861
# # ISIC_0027488 ISIC_0028644 ISIC_0026044

# copy_data("val",val_samples)
# copy_data("train",train_samples)
