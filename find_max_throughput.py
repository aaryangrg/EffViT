
import argparse
import os
from tracemalloc import start
from efficientvit.clscore.data_provider.MiniImageNet import MiniImageNetV2
import time
from efficientvit.cls_model_zoo import create_custom_cls_model 
import torch
import statistics

def select_best_batch_size(model, img_size, max_trials: int = 30) -> int:
    """Returns optimal batch size as measured by throughput (samples / sec)."""

    batch_size = 1
    best_samples_per_sec = 0
    best_batch_size = None
    count = 0
    while count < max_trials :

        try:
            samples_per_sec = evaluate(model, batch_size, img_size)
            print(f"Throughput at batch_size={batch_size}: {samples_per_sec:.5f} samples/s")
            if samples_per_sec < best_samples_per_sec:
                # We assume that once the throughput starts degrading, it won't go up again
                print(f"Throughput dropped at batch {batch_size}")
                break
            best_samples_per_sec = samples_per_sec
            best_batch_size = batch_size
            count += 1
            # double batch size
            batch_size *= 2

        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError):
                print(f"OOM at batch_size={batch_size}")
            else:
                # Not a CUDA error
                raise
            break
        
    if best_batch_size is None:
        print(f"Could not tune batch size, using minimum batch size of {batch_size}")

def evaluate(model, batch_size: int, img_size, total_steps: int = 5) -> float:
    """Evaluates throughput of the given batch size.

    Return:
        Median throughput in samples / sec.
    """

    durations = []
    with torch.no_grad():
        for rep in range(total_steps):
            input = torch.randn(batch_size, 3, img_size, img_size)
            #Including data-transfer time (CPU --> GPU)
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            input = input.cuda()
            _ = model(input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)/1000 
            durations.append(curr_time)

    med_duration_s = statistics.median(durations)
    if med_duration_s == 0.0:
        return float("inf")

    return batch_size / med_duration_s

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/dataset/imagenet/val")
    parser.add_argument("--gpu", type=str, default="all")
    parser.add_argument("-j", "--workers", help="number of workers", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--width_multiplier", type = float, default=1.0)
    parser.add_argument("--depth_multiplier", type = float, default=1.0)
    parser.add_argument("--student_model", type = str, default = None)

    parser.add_argument("--fp16", type = bool, default = False)

    args = parser.parse_args()
    if args.gpu == "all":
        device_list = range(torch.cuda.device_count())
        args.gpu = ",".join(str(_) for _ in device_list)
    else:
        device_list = [int(_) for _ in args.gpu.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model = create_custom_cls_model(args.student_model, False, width_multiplier = args.width_multiplier, depth_multiplier=args.depth_multiplier)

    model.to("cuda:0")
    model.eval()

    # Warm-up iterations
    evaluate(model, 2, args.image_size, total_steps=10)

    # Synchronized Throughput calculation
    select_best_batch_size(model, args.image_size)

if __name__ == "__main__":
    main()



