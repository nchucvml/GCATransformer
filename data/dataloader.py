import os
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset as Ds
from torch.utils.data import DataLoader as Dl
import random


class Co_Dataset(Ds):
    def __init__(
        self,
        dataset_name="Internet_prepro",
        t="train",
        co_input_num=5
    ):
        super().__init__()
        random.seed(9487)

        self.dataset_name = dataset_name
        self.t = t
        self.co_input_num = co_input_num

        self.pixel_mean = np.array(
            [123.675, 116.280, 103.530],
            dtype=np.float32
        )

        self.pixel_std = np.array(
            [58.395, 57.120, 57.375],
            dtype=np.float32
        )

        self.co_file_pths = []
        self.co_class_ids = []
        self.co_class_names = []

        root = os.path.join(dataset_name, t)

        lb_ls = sorted(os.listdir(root))

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

            img_dir = os.path.join(root, lb, "img")
            img_ls = sorted(os.listdir(img_dir))

            for i in range(len(img_ls)):
                co_input_ls = []

                for j in range(co_input_num):
                    img_name = img_ls[(i + j) % len(img_ls)]
                    img_pth = os.path.join(img_dir, img_name)
                    co_input_ls.append(img_pth)

                self.co_file_pths.append(co_input_ls)
                self.co_class_ids.append(class_id)
                self.co_class_names.append(class_name)

    def __len__(self):
        return len(self.co_file_pths)

    def __getitem__(self, index):
        co_file_pth = self.co_file_pths[index]

        class_id = self.co_class_ids[index]
        class_name = self.co_class_names[index]

        batched_data = []

        for img_pth in co_file_pth:
            input_dict = {}

            input_dict["file_name"] = (
                img_pth
                .replace("\\", "/")
                .split("./")[-1]
                .replace("/", "_")
            )

            image = cv.imread(img_pth)

            if image is None:
                raise FileNotFoundError(f"Image not found: {img_pth}")

            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            image = cv.resize(image, (224, 224))
            image = cv.resize(image, (512, 512))

            image = image.astype(np.float32)

            image = np.ascontiguousarray(
                image.transpose(2, 0, 1)
            )

            input_dict["image"] = torch.from_numpy(image)

            img_pth_norm = img_pth.replace("\\", "/")
            mask_pth = img_pth_norm.replace("/img/", "/lb/")

            mask = cv.imread(mask_pth, 0)

            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_pth}")

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

            mask = np.where(mask > 127, 1, 0)

            input_dict["target"] = torch.tensor(
                mask,
                dtype=torch.int64
            )

            # group class label
            input_dict["class"] = torch.tensor(
                class_id,
                dtype=torch.long
            )

            input_dict["class_name"] = class_name

            batched_data.append(input_dict)

        return batched_data


def co_collate_fn(batch):
    flat_batch = []

    for group_idx, group in enumerate(batch):
        for item in group:
            item["group_id"] = group_idx
            flat_batch.append(item)

    return flat_batch


def train_dataloader(
    dataset_name="Internet_prepro",
    batch_size=4,
    co_input_num=5,
    num_workers=4
):
    print("create TrainDataloader for dataset:", dataset_name)

    dataset = Co_Dataset(
        dataset_name=dataset_name,
        t="train",
        co_input_num=co_input_num
    )

    dataloader = Dl(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=co_collate_fn
    )

    print("there are " + str(len(dataloader)) + " batches")
    print("num classes:", len(dataset.class_to_idx))

    return dataloader


def test_dataloader(
    dataset_name="Internet_prepro",
    batch_size=1,
    co_input_num=5,
    num_workers=4
):
    print("create TestDataloader for dataset:", dataset_name)

    dataset = Co_Dataset(
        dataset_name=dataset_name,
        t="test",
        co_input_num=co_input_num
    )

    dataloader = Dl(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=co_collate_fn
    )

    print("there are " + str(len(dataloader)) + " batches")
    print("num classes:", len(dataset.class_to_idx))

    return dataloader


if __name__ == "__main__":
    dataloader = train_dataloader(
        dataset_name="Internet_prepro",
        batch_size=4,
        co_input_num=5
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
