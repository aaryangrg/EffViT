
# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import argparse
import os
import importlib
import sys
sys.path.append('/home/aaryang/experiments/')
from efficientvit.apps import setup
from efficientvit.apps.utils import dump_config, parse_unknown_args
from efficientvit.clscore.trainer import ClsRunConfig
from efficientvit.clscore.trainer.gdino_backbone import GdinoBackboneTrainer
from efficientvit.models.nn.drop import apply_drop_func
from efficientvit.models.efficientvit.dino_backbone import flexible_efficientvit_backbone_swin_t_224_1k
import torch 
from torch.utils.data import DataLoader, DistributedSampler
import json

gdino = importlib.import_module("Open-GDINO")
# gdino_models = importlib.import_module("Open-GDINO.models")
gdino_utils_slconfig = importlib.import_module("Open-GDINO.util.slconfig")
gdino_util_misc = importlib.import_module("Open-GDINO.util.misc")
gdino_datasets = importlib.import_module("Open-GDINO.datasets")

# from Open_GDINO.models.GroundingDINO.groundingdino import build_groundingdino
# from Open_GDINO.datasets import build_dataset
# from Open_GDINO.main import build_model_main

parser = argparse.ArgumentParser()
# Add GDINO args / file paths
parser.add_argument("config", metavar="FILE", help="config file") # Student Model YAML
parser.add_argument("--path", type=str, metavar="DIR", help="run directory") # Path for training outs --> checkpoints + logs
parser.add_argument("--gpu", type=str, default=None)  # used in single machine experiments
parser.add_argument("--manual_seed", type=int, default=0)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--fp16", action="store_true")

# initialization
parser.add_argument("--rand_init", type=str, default="trunc_normal@0.02")
parser.add_argument("--last_gamma", type=float, default=0)

parser.add_argument("--auto_restart_thresh", type=float, default=1.0)
parser.add_argument("--save_freq", type=int, default=1)

# GROUNDING DINO ARGS
parser.add_argument('--config_file', '-c', type=str, required=True)
parser.add_argument('--fix_size', action='store_true')
# dataset parameters
parser.add_argument("--datasets", type=str, required=True, help='path to datasets json')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--pretrain_model_path', help='load from other checkpoint')

def main():
    # parse args
    args, opt = parser.parse_known_args()
    opt = parse_unknown_args(opt)
    print("Args parsed")
    # setup gpu and distributed training
    setup.setup_dist_env(args.gpu)
    print("Distributed env setup")
    # setup path, update args, and save args to path
    os.makedirs(args.path, exist_ok=True)
    dump_config(args.__dict__, os.path.join(args.path, "args.yaml"))

    # Parse GDINO args (config)
    # cfg = gdino_utils_slconfig.SLConfig.fromfile(args.config_file)
    cfg = gdino.util.misc.SLConfig.fromfile(args.config_file)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))
    
    # Extracting val path from dataset file (GDINO)
    with open(args.datasets) as f:
        dataset_meta = json.load(f)
    if args.use_coco_eval:
        args.coco_val_path = dataset_meta["val"][0]["anno"]

    # setup random seed
    setup.setup_seed(args.manual_seed, args.resume)
    print("Random seed setup")
    # setup exp config
    config = setup.setup_exp_config(args.config, recursive=True, opt_args=opt)
    print("Experiment config setup")
    # save exp config
    setup.save_exp_config(config, args.path)

    # setup run config
    run_config = setup.setup_run_config(config, ClsRunConfig)

        # setup model
    effvit_backbone = flexible_efficientvit_backbone_swin_t_224_1k()
    # apply_drop_func(effvit_backbone.stages, config["backbone_drop"])
    effvit_backbone.cuda()

    # Load GDINO model
    model, criterion, postprocessors = gdino.models.groundingdino.build_groundingdino(args)
    model.cuda()
    print("build model, done.")


    # model_without_ddp = model
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    #     model._set_static_graph()
    #     model_without_ddp = model.module
    # SET MODEL TO EVAL
    
    # Dummy input
    # dummy = torch.rand(2,3,1024,1024)
    # dummy = dummy.to("cuda")

    # effvit_backbone.eval()
    # with torch.no_grad() :
    #     print("1x")
    #     effvit_backbone.apply(lambda m: setattr(m, 'width_mult', 1.0))
    #     outs = effvit_backbone(dummy)
    #     for i in range(len(outs)) :
    #         print(outs[i].shape)

    #     print("0.75x")
    #     effvit_backbone.apply(lambda m: setattr(m, 'width_mult', 0.75))
    #     outs = effvit_backbone(dummy)
    #     for i in range(len(outs)) :
    #         print(outs[i].shape)

    #     print("0.50x")
    #     effvit_backbone.apply(lambda m: setattr(m, 'width_mult', 0.50))
    #     outs = effvit_backbone(dummy)
    #     for i in range(len(outs)) :
    #         print(outs[i].shape)
    #     print("0.25x")
    #     effvit_backbone.apply(lambda m: setattr(m, 'width_mult', 0.25))
    #     outs = effvit_backbone(dummy)
    #     for i in range(len(outs)) :
    #         print(outs[i].shape)
    
    # Make this a dataloader somehow??
    dataset_train = gdino_datasets.bbuild_dataset(image_set='train', args=args, datasetinfo=dataset_meta["train"][0])
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,collate_fn=gdino_util_misc.collate_fn, num_workers=8)

    # if args.distributed:
    #     sampler_val = DistributedSampler(dataset_val, shuffle=False)
    #     if not args.eval:
    #         sampler_train = DistributedSampler(dataset_train)
    # else:
    #     sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    #     if not args.eval:
    #         sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # if not args.eval:
    #     batch_sampler_train = torch.utils.data.BatchSampler(
    #         sampler_train, args.batch_size, drop_last=True)
    #     data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
    #                                 collate_fn=utils.collate_fn, num_workers=args.num_workers)
        
    trainer = GdinoBackboneTrainer(
        path=args.path,
        vit_backbone=effvit_backbone,
        dino_backbone=model,
        data_provider=data_loader_train,
        auto_restart_thresh=args.auto_restart_thresh,
    )

    # initialization
    # setup.init_model(
    #     trainer.network,
    #     rand_init=args.rand_init,
    #     last_gamma=args.last_gamma,
    # )

    # prep for training
    # trainer.prep_for_training(run_config, config["ema_decay"], args.fp16)

    # # launch training
    # trainer.train(save_freq=args.save_freq)


if __name__ == "__main__":
    main()
