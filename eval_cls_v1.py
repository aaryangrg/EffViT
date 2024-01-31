# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023


# EVALUATION ON MINI-IMAGENET 10K split

import argparse
import math
import os
from efficientvit.clscore.data_provider.MiniImageNet import MiniImageNet
from efficientvit.clscore.data_provider.CustomDataset import CustomImageDataset
from efficientvit.cls_model_zoo import create_cls_model, create_custom_cls_model , create_flexible_cls_model
from torchpack import distributed as dist

import torch.utils.data
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from efficientvit.apps.utils import AverageMeter
from efficientvit.cls_model_zoo import create_cls_model
from efficientvit.models.nn.norm import reset_bn


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


    parser.add_argument("--reduced_width", type = bool, default = False)
    parser.add_argument("--flexible_width", type = bool, default = False)
    parser.add_argument("--width_multiplier", type = float, default=1.0)
    parser.add_argument("--depth_multiplier", type = float, default=1.0)

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
    dataset = MiniImageNet(args.path, transform = transform, type = "validation")

# Create the data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    if args.flexible_width : 
        model = create_flexible_cls_model(args.model, pretrained = True, weight_url=args.weight_url)
        model.apply(lambda m: setattr(m, 'width_mult', args.width_multiplier))
        # Resetting Batch Norm using train dataset (no extra augmentation)
        bn_reset_dataset = MiniImageNet(args.path, transform = transform, type = "train")
        reset_data_loader = torch.utils.data.DataLoader(
            bn_reset_dataset,
            batch_size=2*args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )
        dloader = []
        for data in reset_data_loader :
            dloader.append(data[0])
        reset_bn(model=model,progress_bar=True, data_loader=dloader, sync = False)
    elif args.reduced_width : 
        model = create_custom_cls_model(args.model, True, weight_url = args.weight_url, width_multiplier = args.width_multiplier, depth_multiplier=args.depth_multiplier)
    else :
        model = create_cls_model(args.model, weight_url=args.weight_url)

    model = torch.nn.DataParallel(model).cuda()
    model.cuda()
       
    model.eval()
    
    top1 = AverageMeter(is_distributed=False)
    top5 = AverageMeter(is_distributed=False)
    with torch.inference_mode():
        with tqdm(total=len(data_loader), desc=f"Eval {args.model} on Mini-ImageNet Validation Set (100 classes)") as t:
            for images, labels in data_loader:
                images, labels = images.cuda(), labels.cuda()
                # compute output
                output = model(images)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
                t.set_postfix(
                    {
                        "top1": top1.avg,
                        "top5": top5.avg,
                        "resolution": images.shape[-1],
                    }
                )
                t.update(1)

    print(f"Top1 Acc={top1.avg:.3f}, Top5 Acc={top5.avg:.3f}")


if __name__ == "__main__":
    main()
