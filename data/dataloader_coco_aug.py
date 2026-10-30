import os
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset as Ds
from torch.utils.data import DataLoader as Dl
import random


class Compose:
    """
    Compose transforms that process image and mask together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for transform in self.transforms:
            image, mask = transform(image, mask)

        return image, mask


class RandomHorizontalFlip:
    """
    Randomly flip image and mask horizontally.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = cv.flip(image, 1)
            mask = cv.flip(mask, 1)

        return image, mask


class RandomBrightnessContrast:
    """
    Randomly change image brightness and contrast.
    The mask is not modified.
    """

    def __init__(
        self,
        brightness_limit=20,
        contrast_limit=0.2,
        p=0.5,
    ):
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            alpha = random.uniform(
                1.0 - self.contrast_limit,
                1.0 + self.contrast_limit,
            )
            beta = random.uniform(
                -self.brightness_limit,
                self.brightness_limit,
            )

            image = image.astype(np.float32)
            image = image * alpha + beta
            image = np.clip(image, 0, 255).astype(np.uint8)

        return image, mask


class RandomResolutionDegradation:
    def __init__(self, min_scale=0.75, max_scale=1.0, p=1.0):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.p = p

    def __call__(self, image, mask):
        if random.random() >= self.p:
            return image, mask

        height, width = image.shape[:2]
        scale = random.uniform(self.min_scale, self.max_scale)

        low_height = max(1, round(height * scale))
        low_width = max(1, round(width * scale))

        image = cv.resize(
            image,
            (low_width, low_height),
            interpolation=cv.INTER_LINEAR,
        )

        image = cv.resize(
            image,
            (width, height),
            interpolation=cv.INTER_LINEAR,
        )

        return image, mask


class RandomGaussianBlur:
    """
    Randomly blur the image.
    The mask is not modified.
    """

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            kernel_size = random.choice([3, 5])

            image = cv.GaussianBlur(
                image,
                (kernel_size, kernel_size),
                sigmaX=0,
            )

        return image, mask


class RandomColorJitter:
    def __init__(
        self,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
        p=0.5,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, image, mask):

        if random.random() >= self.p:
            return image, mask

        img = image.astype(np.float32)

        # brightness
        b = random.uniform(
            1 - self.brightness,
            1 + self.brightness
        )
        img *= b

        # contrast
        c = random.uniform(
            1 - self.contrast,
            1 + self.contrast
        )
        mean = img.mean(axis=(0,1), keepdims=True)
        img = (img - mean) * c + mean

        img = np.clip(img,0,255).astype(np.uint8)

        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.float32)

        # saturation
        s = random.uniform(
            1 - self.saturation,
            1 + self.saturation
        )
        hsv[:,:,1] *= s

        # hue
        h = random.uniform(
            -self.hue*180,
            self.hue*180
        )
        hsv[:,:,0] += h
        hsv[:,:,0] %= 180

        hsv[:,:,1] = np.clip(hsv[:,:,1],0,255)

        img = cv.cvtColor(
            hsv.astype(np.uint8),
            cv.COLOR_HSV2BGR
        )

        return img, mask


class Normalize:
    """
    Normalize an RGB image using the configured ImageNet mean and std.

    Input image range:
        0~255

    Output:
        (image - mean) / std

    The mask is unchanged.
    """

    def __init__(
        self,
        mean=(123.675, 116.280, 103.530),
        std=(58.395, 57.120, 57.375),
    ):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __call__(self, image, mask):
        image = image.astype(np.float32)
        image = (image - self.mean) / self.std

        return image, mask


def get_train_transform():
    """
    Basic augmentation for co-segmentation training.

    Every time Dataset.__getitem__() is called, these transforms
    sample new random parameters.
    """

    return Compose(
        [
            RandomHorizontalFlip(),
            RandomResolutionDegradation(),
            #RandomBrightnessContrast(),
            #Normalize(),
        ]
    )


class Co_Dataset(Ds):
    def __init__(
        self,
        path="G:/CoCo/coco2017/group_mini/",
        t="train",
        co_input_num=5,
        num_per_cat=100,
        s=6666,
    ):
        super().__init__()

        random.seed(s)

        self.path = path
        self.t = t
        self.co_input_num = co_input_num

        # Keep the original external calling method unchanged.
        # Training data automatically uses augmentation.
        # Test data does not use augmentation.
        if self.t == "train":
            self.transform = get_train_transform()
        else:
            self.transform = None

        self.co_file_pths = []
        self.co_cls = []
        self.co_cls_name = []
        self.last_cls = []

        lb_ls = sorted(os.listdir(path + "lb/"))

        self.class_to_idx = {
            lb: idx for idx, lb in enumerate(lb_ls)
        }

        self.idx_to_class = {
            idx: lb for lb, idx in self.class_to_idx.items()
        }

        print("class mapping:")
        for lb, idx in self.class_to_idx.items():
            print(idx, ":", lb)

        for lb in lb_ls:
            class_id = self.class_to_idx[lb]
            class_name = lb

            single_img_dir = path + "img/" + lb + "/single"
            multi_img_dir = path + "img/" + lb + "/multi"

            single_img_ls = sorted(os.listdir(single_img_dir))
            multi_img_ls = sorted(os.listdir(multi_img_dir))

            # random.shuffle(single_img_ls)

            for idx2, img in enumerate(single_img_ls):
                if idx2 >= num_per_cat:
                    break

                co_input_ls = []

                # 第 1 張：anchor / single image
                co_input_ls.append(
                    path + "img/" + lb + "/single/" + img
                )

                # 中間 co_input_num - 2 張：同類 multi images
                for _ in range(co_input_num - 2):
                    co_input_ls.append(
                        path
                        + "img/"
                        + lb
                        + "/multi/"
                        + multi_img_ls[random.randrange(0, len(multi_img_ls))]
                    )

                # 最後一張：有機率塞入其他類別，當 hard negative
                if random.randrange(0, 2) == 0:
                    rlb_idx = random.randrange(0, len(lb_ls))
                    rlb = lb_ls[rlb_idx]

                    rlb_img_dir = path + "img/" + rlb + "/single"
                    rlb_img_ls = sorted(os.listdir(rlb_img_dir))

                    co_input_ls.append(
                        path
                        + "img/"
                        + rlb
                        + "/single/"
                        + rlb_img_ls[random.randrange(0, len(rlb_img_ls))]
                    )

                    # 1 代表最後一張不是同類，target 要設成全 0
                    if rlb == lb:
                        self.last_cls.append(0)
                    else:
                        self.last_cls.append(1)

                else:
                    co_input_ls.append(
                        path
                        + "img/"
                        + lb
                        + "/single/"
                        + single_img_ls[random.randrange(0, len(single_img_ls))]
                    )

                    self.last_cls.append(0)

                self.co_file_pths.append(co_input_ls)
                self.co_cls.append(class_id)
                self.co_cls_name.append(class_name)

    def __len__(self):
        return len(self.co_file_pths)

    def __getitem__(self, index):
        co_file_pth = self.co_file_pths[index]

        class_id = self.co_cls[index]
        class_name = self.co_cls_name[index]

        batched_data = []

        for idx, img_pth in enumerate(co_file_pth):
            input_dict = {}

            input_dict["file_name"] = (
                img_pth
                .replace("\\", "/")
                .split(self.path.replace("\\", "/"))[-1]
                .replace("/", "_")
            )

            image = cv.imread(img_pth)

            if image is None:
                raise FileNotFoundError(f"Image not found: {img_pth}")

            img_pth_norm = img_pth.replace("\\", "/")
            mask_pth = img_pth_norm.replace("/img/", "/lb/")

            mask = cv.imread(mask_pth, 0)

            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_pth}")

            # Keep the original resize pipeline unchanged.
            image = cv.resize(image, (224, 224))
            image = cv.resize(image, (512, 512))

            mask = cv.resize(
                mask,
                (224, 224),
                interpolation=cv.INTER_LINEAR
            )
            mask = cv.resize(
                mask,
                (512, 512),
                interpolation=cv.INTER_LINEAR
            )

            # Apply random training augmentation.
            # Geometry transforms modify image and mask together.
            if self.transform is not None:
                image, mask = self.transform(image, mask)

            image = np.ascontiguousarray(
                image.transpose(2, 0, 1)
            )

            input_dict["image"] = torch.tensor(
                image,
                dtype=torch.float32
            )

            # 如果最後一張是不同類別 negative image，target 設成全 0
            if idx + 1 == self.co_input_num and self.last_cls[index] == 1:
                target = np.zeros_like(mask, dtype=np.int64)
            else:
                target = np.where(mask > 127, 1, 0).astype(np.int64)

            target = np.ascontiguousarray(target)

            input_dict["target"] = torch.tensor(
                target,
                dtype=torch.int64
            )

            # group-level class label
            input_dict["class"] = torch.tensor(
                class_id,
                dtype=torch.long
            )

            input_dict["class_name"] = class_name

            batched_data.append(input_dict)

        return batched_data


def co_collate_fn(batch):
    """
    batch:
        list of groups

    each group:
        list of co_input_num dicts

    output:
        flat list of dicts
        length = batch_size * co_input_num
    """

    flat_batch = []

    for group_idx, group in enumerate(batch):
        for item in group:
            item["group_id"] = group_idx
            flat_batch.append(item)

    return flat_batch


def train_dataloader(
    path="G:/CoCo/coco2017/group_mini/",
    num_per_cat=100,
    s=6666,
    batch_size=4,
    co_input_num=5,
    num_workers=4,
):
    print("create TrainDataloader for dataset:", path)

    dataset = Co_Dataset(
        path=path,
        t="train",
        co_input_num=co_input_num,
        num_per_cat=num_per_cat,
        s=s,
    )

    dataloader = Dl(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=co_collate_fn,
    )

    print("there are " + str(len(dataloader)) + " batches")
    print("num classes:", len(dataset.class_to_idx))

    return dataloader


def test_dataloader(
    path="G:/CoCo/coco2017/group_mini/",
    num_per_cat=100,
    batch_size=1,
    co_input_num=5,
    num_workers=4,
):
    print("create TestDataloader for dataset:", path)

    dataset = Co_Dataset(
        path=path,
        t="test",
        co_input_num=co_input_num,
        num_per_cat=num_per_cat,
    )

    dataloader = Dl(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=co_collate_fn,
    )

    print("there are " + str(len(dataloader)) + " batches")
    print("num classes:", len(dataset.class_to_idx))

    return dataloader


if __name__ == "__main__":
    dataloader = train_dataloader(
        path="G:/CoCo/coco2017/group3/",
        num_per_cat=100,
        batch_size=4,
        co_input_num=5,
    )

    for batch in dataloader:
        print("batch length:", len(batch))
        print("image shape:", batch[0]["image"].shape)
        print("target shape:", batch[0]["target"].shape)
        print("class:", batch[0]["class"])
        print("class_name:", batch[0]["class_name"])

        for i in range(0, len(batch), 5):
            print(
                "group_id:",
                batch[i]["group_id"],
                "class:",
                batch[i]["class"].item(),
                "class_name:",
                batch[i]["class_name"]
            )

        break
