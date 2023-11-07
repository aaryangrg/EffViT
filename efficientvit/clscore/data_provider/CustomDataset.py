import os
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader

class CustomImageDataset(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        super(CustomImageDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.loader = loader
        self.classes = [d.name for d in os.scandir(root) if d.is_dir()]
        self.classes = sorted(self.classes)[:100]  # Taking first 100 classes

        self.samples = []
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root, class_name)
            for file_name in os.listdir(class_path):
                if self.is_image_file(file_name):
                    self.samples.append((os.path.join(class_path, file_name), class_idx))
        print(len(self.samples))

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
