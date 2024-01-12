# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import argparse
import os
from venv import create

from efficientvit.apps import setup
from efficientvit.apps.utils import dump_config, parse_unknown_args
from efficientvit.cls_model_zoo import create_cls_model, create_custom_cls_model
from efficientvit.clscore.data_provider.imagenet_subset import ImageNetDataProviderSubset
from efficientvit.clscore.data_provider import ImageNetDataProvider
from efficientvit.clscore.trainer.cls_kd_trainer import ClsTrainerWithKD
from efficientvit.clscore.trainer import ClsRunConfig
from efficientvit.models.nn.drop import apply_drop_func

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

# Width and Depth customization
parser.add_argument("--reduced_width", type = bool, default = False )
parser.add_argument("--width_multiplier", type = float, default=1.0)
parser.add_argument("--depth_multiplier", type = float, default=1.0)
parser.add_argument("--student_model", type = str, default = None)

def main():
    # parse args
    args, opt = parser.parse_known_args()
    opt = parse_unknown_args(opt)
    # setup gpu and distributed training
    setup.setup_dist_env(args.gpu)
    # setup path, update args, and save args to path
    os.makedirs(args.path, exist_ok=True)
    dump_config(args.__dict__, os.path.join(args.path, "args.yaml"))

    # setup random seed
    setup.setup_seed(args.manual_seed, args.resume)
    # setup exp config
    config = setup.setup_exp_config(args.config, recursive=True, opt_args=opt)
    # save exp config
    setup.save_exp_config(config, args.path)
    run_config = setup.setup_run_config(config, ClsRunConfig)

        # setup model
    if not args.reduced_width :
        if args.student_weights :
            model = create_cls_model(config["net_config"]["name"], True, weight_url = args.student_weights, dropout=config["net_config"]["dropout"])
        else :
            model = create_cls_model(config["net_config"]["name"], False, dropout=config["net_config"]["dropout"])
            print("Distillation from scratch")
    else : 
        print("Using width / depth customization")
        print("Width multiplier : ", args.width_multiplier)
        print("Depth multiplier : ", args.depth_multiplier)
        if args.student_weights :
            model = create_custom_cls_model(args.student_model, True, weight_url = args.student_weights, width_multiplier = args.width_multiplier, depth_multiplier=args.depth_multiplier, dropout=config["net_config"]["dropout"])
        else :
            model = create_custom_cls_model(args.student_model, False, width_multiplier = args.width_multiplier, depth_multiplier=args.depth_multiplier, dropout=config["net_config"]["dropout"])
            print("Distillation from scratch")
    apply_drop_func(model.backbone.stages, config["backbone_drop"])

    total_params = sum(
	    param.numel() for param in model.parameters()
    )
    print(args.student_model, " || ", "w : ", args.width_multiplier, " || ", "d :", args.depth_multiplier, " || params : ",total_params)



if __name__ == "__main__":
    main()
