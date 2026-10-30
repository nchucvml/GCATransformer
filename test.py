"""Evaluate segmentation checkpoints, with optional MSRC postprocessing."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from data.dataloader import test_dataloader

DATASETS = ("PASCAL", "iCoseg", "Internet", "MSRC")


def select_max_region(mask):
    mask = (mask != 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    # OpenCV always assigns label 0 to the background.
    largest_label = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    return (labels == largest_label).astype(np.uint8)


def postprocess_msrc(mask, dilate_iterations):
    mask = select_max_region(mask)
    if dilate_iterations:
        mask = cv2.dilate(
            mask, np.ones((3, 3), dtype=np.uint8), iterations=dilate_iterations
        )
    return mask


def evaluate(model, data_path, args, *, postprocess=False):
    print("Dataset:", data_path)
    dataloader = test_dataloader(
        dataset_name=str(data_path),
        batch_size=args.batch_size,
        co_input_num=args.co_input_num,
        num_workers=args.num_workers,
    )
    accuracies = []
    jaccards = []

    with torch.no_grad():
        for idx, batched_inputs in enumerate(dataloader):
            # print("Batch:", idx)
            outputs = model(batched_inputs)
            if torch.is_tensor(outputs):
                outputs = outputs.detach().cpu().numpy()
            gts = []
            for i in range(0, len(batched_inputs), args.co_input_num):
                gt = batched_inputs[i]["target"]
                if gt.dim() == 3:
                    gt = gt.squeeze(0)
                gts.append(gt.cpu().numpy())
            if len(outputs) != len(gts):
                raise ValueError("Prediction count does not match target group count")

            for pred, gt in zip(outputs, gts):
                if postprocess:
                    pred = postprocess_msrc(pred, args.dilate_iterations)
                if pred.shape != gt.shape:
                    raise ValueError(f"Prediction shape {pred.shape} != target shape {gt.shape}")
                accuracies.append((pred == gt).sum() / pred.size)
                inter = (pred * gt).sum()
                union = pred.sum() + gt.sum() - inter
                jaccards.append(inter / (union + 1e-10))

    if not jaccards:
        raise ValueError(f"No images evaluated in {data_path}")
    # The original scripts called pixel accuracy 'Precision'. Keep the formula.
    print("Pixel Accuracy:", np.mean(accuracies))
    print("Jaccard index:", np.mean(jaccards))
    print("Total evaluated images:", len(jaccards))
    return float(np.mean(accuracies)), float(np.mean(jaccards))


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default="weight/GCA_224_97.2-81.6-98.2-93.2-95.1-75.6-97.6-90.5.sav",
                        help="Whole-model .sav checkpoint")
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parent / "data",
                        help="Directory containing PASCAL, iCoseg, Internet, and MSRC")
    parser.add_argument("--batch-size", type=positive_int, default=7)
    parser.add_argument("--co-input-num", type=positive_int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dilate-iterations", type=int, default=5,
                        help="MSRC dilation iterations (0 keeps only the largest region)")
    return parser


def main(args):
    for name in DATASETS:
        test_dir = args.data_root / name / "test"
        if not test_dir.is_dir():
            raise FileNotFoundError(
                f"Missing dataset directory: {test_dir}. "
                "Place the preprocessed dataset here; see TESTING.md."
            )
    print("Weights:", args.weights)
    print("Evaluation settings:", vars(args))
    # Existing checkpoints contain entire models, rather than state dictionaries.
    model = torch.load(args.weights)
    model.cuda()
    model.eval()
    print("Total params:", sum(p.numel() for p in model.parameters()))
    results = []
    for name in DATASETS:
        accuracy, jaccard = evaluate(
            model, args.data_root / name, args, postprocess=(name == "MSRC")
        )
        results.append((name, accuracy, jaccard))
    print("\nDataset       Pixel Accuracy    Jaccard")
    for name, accuracy, jaccard in results:
        print(f"{name:<13} {accuracy:.6f}          {jaccard:.6f}")


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", UserWarning)
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.num_workers < 0 or args.dilate_iterations < 0:
        parser.error("--num-workers and --dilate-iterations must be nonnegative")
    main(args)
