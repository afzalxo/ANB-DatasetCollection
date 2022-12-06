import os
import argparse
import configparser
import torch
import numpy as np
from collections import OrderedDict

from ..auxiliary.utils import CrossEntropyLabelSmooth
from pytorch_nndct.apis import torch_quantizer, dump_xmodel
from ..models import efficientnet
from ..dataloader import ffcv_dataloader
from ..searchables import searchables
from ..trainval.trainval import infer


def quantize(args, quant_mode):
    float_model = args.build_dir + "/float_model"
    quant_model = args.build_dir + "/quant_model"

    # load trained model
    design = searchables.EfficientNetB0Conf(d=0.5)
    model = efficientnet.efficientnet_b0(design=design, platform="fpga")
    sd = torch.load(os.path.join(float_model, "f_model.pth"),
                    map_location="cpu")
    state_dict = sd["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
    new_state_dict[name] = v

    print("Model Info:")
    print(sd["model_info"])
    print(sd["best_acc_top1"])
    model.load_state_dict(new_state_dict)
    # print(model)

    # override batchsize if in test mode
    if quant_mode == "test":
        batchsize = 1

    rand_in = torch.randn([batchsize, 3, 224, 224])
    quantizer = torch_quantizer(quant_mode, model, (rand_in),
                                output_dir=quant_model)
    quantized_model = quantizer.quant_model
    print("Quantized...")
    # data loader
    _, test_loader, _ = ffcv_dataloader.get_ffcv_loaders(0, args)
    criterion = CrossEntropyLabelSmooth(args.CLASSES,
                                        args.label_smoothing).to(
                                            f'cuda:{args.local_rank}')
    # test_loader, _ = torchvision_dataloader.build_torchvision_loader(dataset_path, batchsize, 4)
    # test_dataset = torchvision.datasets.ImageFolder(dataset_path,
    #                                          transform=test_transform)
    # test_subset = test_dataset#torch.utils.data.Subset(test_dataset, list(range(1)))
    # batchsize=1
    # test_loader = torch.utils.data.DataLoader(test_subset,
    #                                          batch_size=batchsize,
    #                                          shuffle=False)

    # evaluate
    # test(model, device, test_loader)
    # test(quantized_model, device, test_loader)
    infer(test_loader, quantized_model, criterion,
          args, args.report_freq, args.fast)
    # test_all_quantized(quantized_model, device, test_loader)
    # exit(0)

    # export config
    if quant_mode == "calib":
        quantizer.export_quant_config()
    if quant_mode == "test":
        quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)

    return


def run_main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_path", type=str, help="Path to quant config file")
    ap.add_argument(
        "--model_path",
        type=str,
        help="Path to model file",
    )
    ap.add_argument(
        "-q",
        "--quant_mode",
        type=str,
        default="calib",
        choices=["calib", "test"],
        help="Quantization mode (calib or test). Default is calib",
    )
    args = ap.parse_args()
    cfg_path = args.cfg_path
    config = configparser.ConfigParser()
    config.read(cfg_path)
    # Logging
    args.build_dir = config["logging"]["save"]
    args.note = config["logging"]["note"]
    args.report_freq = config["logging"].getint("report_freq")
    args.fast = config["logging"].getboolean("fast")
    args.use_wandb = config["logging"].getboolean("use_wandb")
    # model
    args.label_smoothing = config["model"].getfloat("label_smoothing")
    args.design = config["model"]["design"]
    # dataloaders
    args.val_dataset = config["dataloader"]["val_dataset"]
    args.num_workers = config["dataloader"].getint("num_workers")
    args.in_memory = config["dataloader"].getboolean("in_memory")
    # trainval
    args.val_batch_size = config["trainval"].getint("val_batch_size")
    args.val_resolution = config["trainval"].getint("val_resolution")
    args.lr_tta = config["trainval"].getboolean("lr_tta")
    args.seed = config["trainval"].getint("seed")
    # distributed
    args.distributed = config["distributed"].getboolean("distributed")
    args.cluster = config["distributed"]["cluster"]
    args.port = config["distributed"]["port"]

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
    IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
    DEFAULT_CROP_RATIO = 224 / 256

    args.IMAGENET_MEAN = IMAGENET_MEAN
    args.IMAGENET_STD = IMAGENET_STD
    args.DEFAULT_CROP_RATIO = DEFAULT_CROP_RATIO

    quantize(args, args.quant_mode)

    return


if __name__ == "__main__":
    run_main()
