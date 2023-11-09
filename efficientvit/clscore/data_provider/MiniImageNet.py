import csv
import os
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

DEFAULT_CSV_DIR = "/home/aaryang/experiments/EffViT/efficientvit/mini_csv_files/"

class MiniImageNet(VisionDataset):
    def __init__(self, root, transform=None, type = "train", target_transform=None, loader=default_loader):
        super(MiniImageNet, self).__init__(root, transform=transform, target_transform=target_transform)
        self.loader = loader
        self.classes = [d.name for d in os.scandir(root) if d.is_dir()]
        self.classes = sorted(self.classes) 
        self.type = type
        self.samples = []
    
        csv_dir  = None
        if self.type == "validation" :
            csv_dir = os.path.join(DEFAULT_CSV_DIR, "val.csv")
        else :
            csv_dir = os.path.join(DEFAULT_CSV_DIR, "train.csv")
        
        with open(csv_dir) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            next(csv_reader, None)
            images = {}
            for row in tqdm(csv_reader):
                if row[1] in images.keys():
                    images[row[1]].append(row[0])
                else:
                    images[row[1]] = [row[0]]

            for cls in tqdm(images.keys()):
                for file_name in images[cls] :
                    file_name = cls + "_" + int(file_name.split(".")[0].replace(cls,"")) + ".JPEG"
                    class_idx = self.classes.index(cls)
                    self.samples.append((os.path.join(root, cls, file_name), class_idx))
            print(self.type, " : ", len(self.samples))
       

    def __getitem__(self, index):
        path, target = self.samples[index]
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
