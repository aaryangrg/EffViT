# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023


# EVALUATION ON IMAGENET 100 class subset

import argparse
import math
import os
from efficientvit.clscore.data_provider.MiniImageNet import MiniImageNetV2
from torch.profiler import profile as profiler, record_function, ProfilerActivity
from thop import profile

import torch.utils.data
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from efficientvit.apps.utils import AverageMeter
from efficientvit.cls_model_zoo import create_cls_model, create_custom_cls_model 


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list[torch.Tensor]:
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/dataset/imagenet/val")
    parser.add_argument("--gpu", type=str, default="all")
    parser.add_argument("--batch_size", help="batch size per gpu", type=int, default=50)
    parser.add_argument("-j", "--workers", help="number of workers", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--crop_ratio", type=float, default=0.95)
    parser.add_argument("--model", type=str)
    parser.add_argument("--weight_url", type=str, default=None)


    parser.add_argument("--reduced_width", type = bool, default = False )
    parser.add_argument("--width_multiplier", type = float, default=1.0)
    parser.add_argument("--depth_multiplier", type = float, default=1.0)
    parser.add_argument("--student_model", type = str, default = "b1_custom")

    args = parser.parse_args()
    if args.gpu == "all":
        device_list = range(torch.cuda.device_count())
        args.gpu = ",".join(str(_) for _ in device_list)
    else:
        device_list = [int(_) for _ in args.gpu.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.batch_size = args.batch_size * max(len(device_list), 1)

    transform = transforms.Compose([
        transforms.Resize(
            int(math.ceil(args.image_size / args.crop_ratio)), interpolation=InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create the dataset
    dataset = MiniImageNetV2(args.path, transform = transform, type = "validation")

# Create the data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    input = torch.randn(1, 3, 224, 224)
    input.to(dtype=torch.float16)
    input.to("cuda:0")

    model = create_custom_cls_model(args.student_model, False, width_multiplier = args.width_multiplier, depth_multiplier=args.depth_multiplier)
        
    # model = torch.nn.DataParallel(model).cuda()
    model.to("cuda:0")
    model.eval()

    # # Print GPU memory summary
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))

    with torch.inference_mode():
        with tqdm(total=len(data_loader)) as t:
            for images, labels in data_loader:

                # Batch recorded
                images, labels = images.cuda(), labels.cuda()

                with profiler(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
                    model(images)

                print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
                
                # MACS calculation & Params (single image)
                macs, params = profile(model, input)
                print(f"MACSs: {macs}, Params: {params}")


if __name__ == "__main__":
    main()


