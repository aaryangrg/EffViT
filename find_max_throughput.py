
import argparse
import os
from tracemalloc import start
from efficientvit.clscore.data_provider.MiniImageNet import MiniImageNetV2
import time
from efficientvit.cls_model_zoo import create_custom_cls_model 
import torch
import statistics

def select_best_batch_size(model, img_size, max_trials: int = 30, fp16 = False) -> int:
    """Returns optimal batch size as measured by throughput (samples / sec)."""

    batch_size = 1
    best_samples_per_sec = 0
    best_batch_size = None
    count = 0
    while count < max_trials :

        try:
            samples_per_sec = evaluate(model, batch_size, img_size, fp16 = fp16)
            print(f"Throughput at batch_size={batch_size}: {samples_per_sec:.5f} samples/s")
            if samples_per_sec > best_samples_per_sec :
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

    print("Discovering optimal batch-size")
    best_throughput, best_batch = binary_search(model, batch_size // 2, batch_size, img_size, fp16 = fp16)

    if best_batch_size is None:
        print(f"Could not tune batch size, using minimum batch size of {batch_size}")
        return
    
    true_max_samples_per_sec = max(best_throughput, samples_per_sec)
    print("Maximum : ", true_max_samples_per_sec)

def binary_search(model, low, hi, img_size, fp16 = False) :
    best_throughput, best_batch_size = 0, 0
    while low <= hi :
        mid = (hi + low) // 2
        try :
            cur_throughput = evaluate(model, mid, img_size, fp16 = fp16)
            if cur_throughput > best_throughput :
                best_throughput = cur_throughput
                best_batch_size = mid
            low = mid + 1
        except RuntimeError as e :
            if "CUDA out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError):
                print(f"OOM at batch_size={mid}")
            hi = mid - 1
    return best_throughput, best_batch_size
        

def evaluate(model, batch_size: int, img_size, total_steps: int = 10, fp16 = False) -> float:
    """Evaluates throughput of the given batch size.

    Return:
        Median throughput in samples / sec.
    """
    if fp16 :
        model = model.half()
    durations = []
    with torch.no_grad():
        for rep in range(total_steps):
            input = torch.randn(batch_size, 3, img_size, img_size)
            #Including data-transfer time (CPU --> GPU)
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            input.to("cuda:0")
            if fp16 :
                input = input.half()
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
    evaluate(model, 2, args.image_size, total_steps=20, fp16=args.fp16)

    # Synchronized Throughput calculation
    select_best_batch_size(model, args.image_size, fp16=args.fp16)

if __name__ == "__main__":
    main()



