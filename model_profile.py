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
    parser.add_argument("--find_macs", type = bool, default = True)
    parser.add_argument("--profile", type = bool, default = True)

    args = parser.parse_args()
    if args.gpu == "all":
        device_list = range(torch.cuda.device_count())
        args.gpu = ",".join(str(_) for _ in device_list)
    else:
        device_list = [int(_) for _ in args.gpu.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    input = torch.randn(1, 3, args.image_size, args.image_size)
    input.to(dtype=torch.float16)
    input = input.cuda()
    model = create_custom_cls_model(args.student_model, False, width_multiplier = args.width_multiplier, depth_multiplier=args.depth_multiplier)
        
    model.to("cuda:0")
    model.eval()

    # # Print GPU memory summary
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    if args.profile :
        for i in range(5) :
            input = input.cuda()
            # Batch recorded
            with profiler(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
                model(input)
            print(prof.key_averages().table(sort_by="self_cuda_time_total"))
        #MACS calculation & Params (single image)
    # if args.find_macs : 
    #     macs, params = profile(model, inputs = (input,))
    #     print(f"MACSs: {macs}, Params: {params}")

if __name__ == "__main__":
    main()


