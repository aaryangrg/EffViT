# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import argparse
import os
from venv import create

from efficientvit.apps import setup
from efficientvit.apps.utils import dump_config, parse_unknown_args
from efficientvit.cls_model_zoo import create_cls_model
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

# Parent model
parser.add_argument("--parent_weights_url", type =str, default = "Parent Weights")
parser.add_argument("--parent_model", type =str, default = "Parent Model")
parser.add_argument("--use_subset", type = bool , default = True)

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

    # setup data provider
    # if args.use_subset :
    #     data_provider = setup.setup_data_provider(config, [ImageNetDataProviderSubset], is_distributed=True)
    # else :
    data_provider = setup.setup_data_provider(config, [ImageNetDataProvider], is_distributed=True)

    # setup run config
    run_config = setup.setup_run_config(config, ClsRunConfig)

    # setup model
    model = create_cls_model(config["net_config"]["name"], False, dropout=config["net_config"]["dropout"])
    apply_drop_func(model.backbone.stages, config["backbone_drop"])

    print("student model created")

    p_model = create_cls_model(args.parent_model, args.parent_weights_url)

    print("parent model created")
    # setup trainer
    trainer = ClsTrainerWithKD(
        path=args.path,
        model=model,
        p_model = p_model,
        data_provider=data_provider,
        auto_restart_thresh=args.auto_restart_thresh,
    )
    # initialization
    setup.init_model(
        trainer.network,
        rand_init=args.rand_init,
        last_gamma=args.last_gamma,
    )

    # prep for training
    trainer.prep_for_training(run_config, config["ema_decay"], args.fp16)

    # resume
    if args.resume:
        trainer.load_model()
        # if args.use_subset :
        #     trainer.data_provider = setup.setup_data_provider(config, [ImageNetDataProviderSubset], is_distributed=True)
        # else :
        trainer.data_provider = setup.setup_data_provider(config, [ImageNetDataProvider], is_distributed=True)
    else:
        trainer.sync_model()

    # launch training
    trainer.train(save_freq=args.save_freq)


if __name__ == "__main__":
    main()
