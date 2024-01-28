
import argparse
import os
from tracemalloc import start
from efficientvit.clscore.data_provider.MiniImageNet import MiniImageNetV2
import time
from efficientvit.cls_model_zoo import create_custom_cls_model 
import torch
import statistics

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

    def select_best_batch_size(
            model,
            max_batch_size,
            max_trials: int = 20,
        ) -> int:
            """Returns optimal batch size as measured by throughput (samples / sec)."""

            batch_size = 1
            best_samples_per_sec = 0
            best_batch_size = None
            count = 0
            while count < max_trials and _is_valid_batch_size(batch_size):
                if is_coordinator:
                    logger.info(f"Exploring batch_size={batch_size}")
                gc.collect()

                try:
                    samples_per_sec = self.evaluate(
                        batch_size, total_steps=TOTAL_STEPS, global_max_sequence_length=global_max_sequence_length
                    )
                    if is_coordinator:
                        logger.info(f"Throughput at batch_size={batch_size}: {samples_per_sec:.5f} samples/s")
                    if samples_per_sec < best_samples_per_sec:
                        # We assume that once the throughput starts degrading, it won't go up again
                        if is_coordinator:
                            logger.info(f"Throughput decrease at batch_size={batch_size}")
                        break

                    best_samples_per_sec = samples_per_sec
                    best_batch_size = batch_size
                    count += 1

                    # double batch size
                    batch_size *= 2
                except RuntimeError as e:
                    # PyTorch only generates Runtime errors for CUDA OOM.
                    gc.collect()
                    if "CUDA out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError):
                        if is_coordinator:
                            logger.info(f"OOM at batch_size={batch_size}")
                    else:
                        # Not a CUDA error
                        raise
                    break

            # Ensure that some batch size is found.
            # `best_batch_size` can be None if the first batch size is invalid.
            if best_batch_size is None:
                if is_coordinator:
                    logger.info(f"Could not tune batch size, using minimum batch size of {MIN_POSSIBLE_BATCH_SIZE}")
                best_batch_size = MIN_POSSIBLE_BATCH_SIZE

            if is_coordinator:
                logger.info(f"Selected batch_size={best_batch_size}")
            return best_batch_size

    def evaluate(
        self, batch_size: int, total_steps: int = 5, global_max_sequence_length: Optional[int] = None
    ) -> float:
        """Evaluates throughput of the given batch size.

        Return:
            Median throughput in samples / sec.
        """
        durations = []
        for _ in range(total_steps):
            self.reset()
            start_ts = time.time()
            self.step(batch_size, global_max_sequence_length=global_max_sequence_length)
            durations.append(time.time() - start_ts)

        med_duration_s = statistics.median(durations)
        if med_duration_s == 0.0:
            return float("inf")

        return batch_size / med_duration_s
    
if __name__ == "__main__":
    main()



