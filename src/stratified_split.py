import os
import shutil
from sklearn.model_selection import train_test_split


class StratifiedSplit:
    def __init__(
        self, 
        orig_path: str = "auto",
        train_path: str = "auto_train",
        val_path: str = "auto_val",
        val_size: float = 0.2
    ):
        images = []
        classes = []

        for image_name in os.listdir(orig_path):
            class_name = image_name.split("_")[1]
            images.append(image_name)
            classes.append(class_name)
        
        train_images, val_images, _, _ = train_test_split(images, classes, stratify=classes, test_size=val_size)

        if not os.path.exists(train_path):
            os.makedirs(train_path)
        for image_name in train_images:
            shutil.copy(os.path.join(orig_path, image_name), os.path.join(train_path, image_name))

        if not os.path.exists(val_path):
            os.makedirs(val_path)
        for image_name in val_images:
            shutil.copy(os.path.join(orig_path, image_name), os.path.join(val_path, image_name))
