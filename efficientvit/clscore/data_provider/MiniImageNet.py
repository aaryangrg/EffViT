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
    
        
        # if self.type == "validation" :
        #     csv_dir = os.path.join(DEFAULT_CSV_DIR, "test.csv")
        # else :
        #     csv_dir = os.path.join(DEFAULT_CSV_DIR, "trainval.csv")

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
