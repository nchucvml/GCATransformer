from typing import Tuple

import torch
from torch import nn
from torch.nn.functional import cross_entropy

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import ImageList

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .group_attention import Group_Att


@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
    ):
        super().__init__()

        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion

        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata

        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility

        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference

        self.register_buffer(
            "pixel_mean",
            torch.Tensor(pixel_mean).view(-1, 1, 1),
            False,
        )
        self.register_buffer(
            "pixel_std",
            torch.Tensor(pixel_std).view(-1, 1, 1),
            False,
        )

        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        # co_input_num = 5
        self.group_att = nn.ModuleList([
            Group_Att() #Group_Att(group_size=5)
        ])

        self.soft = nn.Softmax(dim=2)
        self.sig = nn.Sigmoid()

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}

            for i in range(dec_layers - 1):
                aux_weight_dict.update({
                    k + f"_{i}": v for k, v in weight_dict.items()
                })

            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def prepare_images_and_labels(self, batched_inputs):
        images = []
        labels = []

        for x in batched_inputs:
            image = x["image"].to(self.device).float()

            # 避免 DataLoader 多包一層 batch 維度
            if image.dim() == 4:
                image = image.squeeze(0)

            image = (image - self.pixel_mean) / self.pixel_std
            images.append(image)

            if "target" in x:
                target = x["target"].to(self.device).long()

                if target.dim() == 3:
                    target = target.squeeze(0)

                labels.append(target)

        images = ImageList.from_tensors(
            images,
            self.size_divisibility,
        )

        if len(labels) > 0:
            labels = torch.stack(labels, dim=0)
        else:
            labels = None

        return images, labels

    def forward(self, batched_inputs):
        """
        batched_inputs:
            len = batch_size * group_size

        example:
            batch_size = 4
            group_size = 5
            len(batched_inputs) = 20
        """

        images, labels = self.prepare_images_and_labels(batched_inputs)

        features = self.backbone(images.tensor)
        del images

        features = self.group_att[0](features)

        outputs = self.sem_seg_head(features)
        del features

        mask_cls = self.soft(outputs["pred_logits"])
        mask_pred = self.sig(outputs["pred_masks"])

        outputs = torch.einsum(
            "bqc,bqhw->bchw",
            mask_cls,
            mask_pred,
        )

        outputs = self.up(outputs)
        outputs = self.up(outputs)

        if self.training:
            seg_loss = cross_entropy(
                outputs,
                labels,
            )
            return seg_loss
        else:

            group_size = 5

            # outputs: [B*5, C, H, W]

            pred = torch.argmax(outputs, dim=1)  # [B*5, H, W]

            # 每個 group 只取第 0 張

            # 也就是 index: 0, 5, 10, 15, ...

            pred = pred[::group_size]  # [B, H, W]

            return pred.cpu().detach().numpy()