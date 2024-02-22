
# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import argparse
import os
from efficientvit.apps import setup
from efficientvit.apps.utils import dump_config, parse_unknown_args
from efficientvit.clscore.trainer import ClsRunConfig
from efficientvit.models.nn.drop import apply_drop_func
from efficientvit.models.efficientvit.dino_backbone import flexible_efficientvit_backbone_swin_t_224_1k
import torch 

parser = argparse.ArgumentParser()
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
    effvit_backbone.cuda()
    # Dummy input
    dummy = torch.rand(2,3,1024,1024)
    dummy = dummy.to("cuda")
    outs = effvit_backbone(dummy)
    for i in range(len(outs)) :
        print(outs[i].shape)
    
    # apply_drop_func(model.backbone.stages, config["backbone_drop"])

    # setup.init_model(
    #     trainer.network,
    #     rand_init=args.rand_init,
    #     last_gamma=args.last_gamma,
    # )

    # prep for training
    # trainer.prep_for_training(run_config, config["ema_decay"], args.fp16)

    # resume
    # if args.resume:
    #     trainer.load_model()
    #     data_provider = setup.setup_data_provider(config, [ImageNetDataProviderSubset], is_distributed=True)
    # else:
    #     trainer.sync_model()

    # # launch training
    # trainer.train(save_freq=args.save_freq)


if __name__ == "__main__":
    main()
