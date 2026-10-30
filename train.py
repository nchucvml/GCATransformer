from detectron2.engine import default_argument_parser, default_setup
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config

from detectron2.modeling import build_model

import copy
import itertools
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Set

from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.checkpoint import DetectionCheckpointer

import torch
import numpy as np

from data.dataloader import test_dataloader
from data.dataloader_coco_aug import train_dataloader


def patch_old_model(model):
    if not hasattr(model, "group_size"):
        model.group_size = 5

    if hasattr(model, "group_att"):
        for m in model.group_att:
            if not hasattr(m, "group_size"):
                m.group_size = 5

    return model


def load_checkpoint(model, pkl):
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(pkl)
    return checkpointer.model


def build_optimizer(cfg, model):
    weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
    weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

    defaults = {
        "lr": cfg.SOLVER.BASE_LR,
        "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
    }

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()

    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue

            if value in memo:
                continue

            memo.add(value)
            hyperparams = copy.copy(defaults)

            if "backbone" in module_name:
                hyperparams["lr"] = (
                    hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                )

            if (
                "relative_position_bias_table" in module_param_name
                or "absolute_pos_embed" in module_param_name
            ):
                print(module_param_name)
                hyperparams["weight_decay"] = 0.0

            if isinstance(module, norm_module_types):
                hyperparams["weight_decay"] = weight_decay_norm

            if isinstance(module, torch.nn.Embedding):
                hyperparams["weight_decay"] = weight_decay_embed

            params.append({
                "params": [value],
                **hyperparams,
            })

    def maybe_add_full_model_gradient_clipping(optim):
        clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE

        enable = (
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED
            and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
            and clip_norm_val > 0.0
        )

        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(
                    *[x["params"] for x in self.param_groups]
                )
                torch.nn.utils.clip_grad_norm_(
                    all_params,
                    clip_norm_val,
                )
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optim

    optimizer_type = cfg.SOLVER.OPTIMIZER

    if optimizer_type == "SGD":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
            params,
            cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
        )

    elif optimizer_type == "ADAMW":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
            params,
            cfg.SOLVER.BASE_LR,
        )

    elif optimizer_type == "ADAM":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.Adam)(
            params,
            cfg.SOLVER.BASE_LR,
        )

    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")

    if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
        optimizer = maybe_add_gradient_clipping(
            cfg,
            optimizer,
        )

    return optimizer


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = str(Path(args.output_dir) / args.experiment_name)
    cfg.freeze()

    default_setup(cfg, args)

    return cfg


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group["lr"]


def load_pretrained_partial(model, weight_path):
    load_options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_options["weights_only"] = False
    ckpt = torch.load(weight_path, **load_options)

    if isinstance(ckpt, torch.nn.Module):
        ckpt = patch_old_model(ckpt)
        ckpt_state = ckpt.state_dict()

    elif isinstance(ckpt, dict) and "model" in ckpt:
        ckpt_state = ckpt["model"]

    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt_state = ckpt["state_dict"]

    else:
        ckpt_state = ckpt

    model = patch_old_model(model)
    model_state = model.state_dict()

    load_state = {}
    skipped = []

    for k, v in ckpt_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            load_state[k] = v
        else:
            skipped.append(k)

    model_state.update(load_state)
    model.load_state_dict(model_state, strict=False)
    model = patch_old_model(model)

    print("Loaded params:", len(load_state))
    print("Skipped params:", len(skipped))

    return model


def evaluate(
    model,
    t_dataloader,
    co_input_num=5,
):
    model.eval()

    all_p = []
    all_j = []

    with torch.no_grad():
        for t_batched_inputs in t_dataloader:
            outputs = model(t_batched_inputs)

            gts = []

            for i in range(0, len(t_batched_inputs), co_input_num):
                gt = t_batched_inputs[i]["target"]

                if gt.dim() == 3:
                    gt = gt.squeeze(0)

                gts.append(gt.cpu().numpy())

            gts = np.stack(gts, axis=0)

            for b in range(outputs.shape[0]):
                pred = outputs[b]
                gt = gts[b]

                precision = (pred == gt).sum() / pred.size

                inter = (pred * gt).sum()
                union = pred.sum() + gt.sum() - inter
                jaccard = inter / (union + 1e-10)

                all_p.append(precision)
                all_j.append(jaccard)

    return np.mean(all_p), np.mean(all_j)


def get_item(loss_dict, key):
    if key in loss_dict:
        return loss_dict[key].item()
    return 0.0


def main(args):
    epoches = args.epochs
    agpi = args.accumulation_steps
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    co_input_num = args.co_input_num
    cfg = setup(args)
    output_dir = Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    model = build_model(cfg)

    if cfg.MODEL.WEIGHTS:
        if cfg.MODEL.WEIGHTS.lower().endswith(".sav"):
            model = load_pretrained_partial(model, cfg.MODEL.WEIGHTS)
        else:
            model = load_checkpoint(model, cfg.MODEL.WEIGHTS)

    model = patch_old_model(model)
    model.cuda()

    optimizer = build_optimizer(cfg, model)

    print("start lr:", get_lr(optimizer))
    print("optimizer:", cfg.SOLVER.OPTIMIZER)
    print("train_batch_size:", train_batch_size)
    print("test_batch_size:", test_batch_size)
    print("co_input_num:", co_input_num)
    print("agpi:", agpi)

    model.train()

    ag = 0
    itr = 0
    maxacc = 0
    maxiou = 0

    ag_loss_total = 0.0
    ag_loss_seg = 0.0

    t_dataloader = test_dataloader(
        dataset_name=args.test_data,
        batch_size=test_batch_size,
        co_input_num=co_input_num,
        num_workers=args.num_workers,
    )

    for epoch in range(epoches):
        dataloader = train_dataloader(
            path=args.train_data,
            num_per_cat=args.num_per_cat,
            s=epoch,
            batch_size=train_batch_size,
            co_input_num=co_input_num,
            num_workers=args.num_workers,
        )

        print("epoch:", epoch, "/", epoches)

        for idx, batched_inputs in enumerate(dataloader):
            if ag == 0:
                optimizer.zero_grad()

                ag_loss_total = 0.0
                ag_loss_seg = 0.0

            loss = model(batched_inputs)

            if not torch.is_tensor(loss):
                raise TypeError(f"Expected tensor loss, but got {type(loss)}")

            (loss / agpi).backward()

            ag_loss_total += loss.item()
            ag_loss_seg += loss.item()

            ag += 1

            if ag == agpi:
                itr += 1

                optimizer.step()
                optimizer.zero_grad()

                print(
                    "-------------------------------------",
                    "epoch:", epoch,
                    "idx:", idx,
                    "itr:", itr,

                    "loss_total:", round(ag_loss_total / agpi, 4),
                    "loss_seg:", round(ag_loss_seg / agpi, 4),
                )

                ag = 0

                if itr % 1000 == 0 and itr > 999:
                    test_p, test_j = evaluate(
                        model,
                        t_dataloader,
                        co_input_num=co_input_num,
                    )

                    print(
                        "Epoch:",
                        epoch,
                        "itr:",
                        itr,
                        "Test Precision:",
                        test_p,
                        "Jaccard index:",
                        test_j,
                    )

                    if test_p > maxacc:
                        print("new best precision")
                        maxacc = test_p
                        torch.save(
                            model,
                            output_dir / "best_precision.sav",
                        )

                    if test_j > maxiou:
                        print("new best iou")
                        maxiou = test_j
                        torch.save(
                            model,
                            output_dir / "best_iou.sav",
                        )

                    model.train()

        if ag > 0:
            itr += 1

            optimizer.step()
            optimizer.zero_grad()

            denom = ag

            print(
                "epoch end step",
                "epoch:", epoch,
                "itr:", itr,

                "loss_total:", round(ag_loss_total / denom, 4),
                "loss_seg:", round(ag_loss_seg / denom, 4),
            )

            ag = 0

        torch.save(model, output_dir / "last.sav")

    return


def positive_int(value):
    value = int(value)
    if value < 1:
        raise ValueError("must be a positive integer")
    return value


def build_argument_parser():
    parser = default_argument_parser()
    parser.set_defaults(config_file="configs/maskformer2_swin_tiny_bs16_50ep.yaml")
    parser.add_argument("--train-data", default="G:/CoCo/coco2017/group_ufo/",
                        help="Training dataset directory (default: %(default)s)")
    parser.add_argument("--test-data", default="data/iCoSeg",
                        help="Evaluation dataset directory (default: %(default)s)")
    parser.add_argument("--output-dir", default="outputs", help="Root directory for runs")
    parser.add_argument("--experiment-name", default="swin_tiny_baseline",
                        help="Run directory name (default: %(default)s)")
    parser.add_argument("--epochs", type=positive_int, default=5)
    parser.add_argument("--accumulation-steps", type=positive_int, default=8)
    parser.add_argument("--train-batch-size", type=positive_int, default=2)
    parser.add_argument("--test-batch-size", type=positive_int, default=1)
    parser.add_argument("--co-input-num", type=positive_int, default=5)
    parser.add_argument("--num-per-cat", type=positive_int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", UserWarning)

    parser = build_argument_parser()
    args = parser.parse_args()
    if not args.config_file:
        parser.error("--config-file is required")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    if args.experiment_name in {".", ".."} or any(
        char in args.experiment_name for char in '/\\:'
    ):
        parser.error("--experiment-name must be a single directory name")
    print("Command Line Args:", args)

    main(args)
