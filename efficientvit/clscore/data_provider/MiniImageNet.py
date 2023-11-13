import csv
import os
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from tqdm import tqdm
from torch.nn.functional import one_hot

DEFAULT_CSV_DIR = "/home/aaryang/experiments/EffViT/efficientvit/mini_csv_files/"

class MiniImageNet(VisionDataset):
    def __init__(self, root, transform=None, type = "train", target_transform=None, loader=default_loader):
        super(MiniImageNet, self).__init__(root, transform=transform, target_transform=target_transform)
        self.loader = loader
        self.classes = [d.name for d in os.scandir(root) if d.is_dir()]
        self.classes = sorted(self.classes) 
        self.type = type
        self.samples = []

        csv_dir  = os.path.join(DEFAULT_CSV_DIR, "allimages.csv")
        # 600 samples / class x 100 classes
        with open(csv_dir) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            next(csv_reader, None)
            images = {}
            count = 0
            if type == "train" :
                for row in tqdm(csv_reader):
                    if count % 6 != 0 :
                        if len(row) == 2 and row[1] in images.keys():
                            images[row[1]].append(row[0])
                        else:
                            if len(row) == 2 :
                                images[row[1]] = [row[0]]
                    count += 1
            else :
                for row in tqdm(csv_reader):
                    if count % 6 == 0 :
                        if len(row) == 2 and row[1] in images.keys():
                            images[row[1]].append(row[0])
                        else:
                            if len(row) == 2 :
                                images[row[1]] = [row[0]]
                    count += 1
            
            for cls in tqdm(images.keys()):
                for file_name in images[cls] :
                    class_path = os.path.join(root, cls)
                    class_idx = self.classes.index(cls)
                    self.samples.append((os.path.join(class_path, file_name), class_idx))
            print(self.type, " : ", len(self.samples))
       

    def __getitem__(self, index):
        path, target = self.samples[index]
        # One-hot the target --> happens in before_step --> label_smooth function
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in [".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".tif", ".JPEG"])


class MiniImageNetV2(VisionDataset):
    # Mini-ImageNet (60K) --> all used for train (100 classes x 600 img)
    # Validation --> ImageNet validation files on Mini-ImageNet classes
    def __init__(self, root, transform=None, type = "train", target_transform=None, loader=default_loader):
        super(MiniImageNetV2, self).__init__(root, transform=transform, target_transform=target_transform)
        self.loader = loader
        self.classes = [d.name for d in os.scandir(root) if d.is_dir()]
        self.classes = sorted(self.classes) 
        self.type = type
        self.samples = []
        images = {}

        if self.type == "train" :
            csv_dir  = os.path.join(DEFAULT_CSV_DIR, "allimages.csv")
            # 600 samples / class x 100 classes
            with open(csv_dir) as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',')
                next(csv_reader, None)
                if type == "train" :
                    for row in tqdm(csv_reader):
                        if len(row) == 2 and row[1] in images.keys():
                            images[row[1]].append(row[0])
                        else:
                            if len(row) == 2 :
                                images[row[1]] = [row[0]]
            
            for cls in tqdm(images.keys()):
                for file_name in images[cls] :
                    class_path = os.path.join(root, cls)
                    class_idx = self.classes.index(cls)
                    self.samples.append((os.path.join(class_path, file_name), class_idx))
        else :
            class_dir = os.path.join(DEFAULT_CSV_DIR, "class_list.txt")
            with open(class_dir) as class_file :
                clses = class_file.readlines()
                clses = [cls.strip() for cls in clses]
            
            self.samples = []
            for class_name in clses:
                class_path = os.path.join(root, class_name)
                class_idx = self.classes.index(class_name)
                for file_name in os.listdir(class_path):
                    if self.is_image_file(file_name):
                        self.samples.append((os.path.join(class_path, file_name), class_idx))
        
        print(self.type, " : ", len(self.samples))

    def __getitem__(self, index):
        path, target = self.samples[index]
        # One-hot the target --> happens in before_step --> label_smooth function
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in [".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".tif", ".JPEG"])
