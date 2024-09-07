import os
import shutil
from torch.utils.data import Dataset
from PIL import Image


class CarModelDataset(Dataset):
    def __init__(
        self, 
        root_dir: str = "auto", 
        aug_dir: str = "auto_aug",
        aug_transform = None, 
        transform = None, 
        num_augmentations: int = 3
    ):
        self.root_dir = root_dir
        self.aug_dir = aug_dir
        self.aug_transform = aug_transform
        self.transform = transform
        self.num_augmentations = num_augmentations
        self.images = []
        self.labels = []
        self.class_names = dict()

        if aug_transform:
            work_dir = aug_dir
            self.__augment_images()
        else:
            work_dir = root_dir

        for image_name in os.listdir(work_dir):
            class_name = image_name.split("_")[1]
            if class_name not in self.class_names.keys():
                self.class_names[class_name] = len(self.class_names)
            self.images.append(os.path.join(work_dir, image_name))
            self.labels.append(self.class_names[class_name])

    def __augment_images(self):
        if not os.path.exists(self.aug_dir):
            os.makedirs(self.aug_dir)
        for image_name in os.listdir(self.root_dir):
            image = Image.open(os.path.join(self.root_dir, image_name)).convert('RGB')
            name, ext = os.path.splitext(image_name)

            shutil.copy(os.path.join(self.root_dir, image_name), os.path.join(self.aug_dir, image_name))

            for i in range(self.num_augmentations):
                aug_image = self.aug_transform(image)
                aug_image.save(os.path.join(self.aug_dir, f"{name}_aug_{i}{ext}"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
