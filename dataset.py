# -*- coding: utf-8 -*-
"""
Modified dataset.py
- make_dataset() unchanged — your existing .npy files still work, no need to regenerate.
- make_data() now applies online augmentation when stage='train'.

Augmentation strategy:
- Per-instance (each cell in the bag gets its own random transform).
  This is appropriate because cells are sampled independently and
  rotation/flip don't change cell-level pathology.
- Bag-level instance dropout (randomly drops 0-3 instances per bag during training)
  improves attention-pooling robustness.
- Stays in [0, 1] tensor space, no PIL conversions.
"""
import os
import random
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from glob import glob


# ============================================================
# make_dataset() — offline preprocessor, UNCHANGED from your version
# ============================================================
def make_dataset(rootdir, seq_dir, seq=None, extension=None):
    class_name = ['5', '6', '7', '8', '9']
    seeds = [111, 222]
    data_dir = os.listdir(rootdir)
    count = 1
    for data_dir_name in data_dir:
        data_imgs_dir = os.path.join(rootdir, data_dir_name)
        if data_dir_name == class_name[0]: label = 0
        if data_dir_name == class_name[1]: label = 1
        if data_dir_name == class_name[2]: label = 2
        if data_dir_name == class_name[3]: label = 3
        if data_dir_name == class_name[4]: label = 4
        print('class_name:{},label:{}'.format(data_dir_name, label))
        label = torch.tensor(label).long()

        for sample_dir_name in os.listdir(data_imgs_dir):
            sample_dir = os.path.join(data_imgs_dir, sample_dir_name)
            image_path = sample_dir + '/*.png'
            img_list = glob(image_path)
            for extension_num in range(extension):
                np.random.seed(seeds[0])
                for i in range(10000):
                    np.random.shuffle(img_list)

                img_num = len(img_list)
                N = int(np.floor(img_num / seq))
                for i in range(N):
                    dataset = []
                    for index in range(seq):
                        tmp_img = Image.open(img_list[seq * i + index])
                        tmp_img = tmp_img.resize((512, 512), Image.ANTIALIAS)
                        tmp_img = np.array(tmp_img)
                        tmp_img = tmp_img / float(255)
                        tmp_img = torch.from_numpy(tmp_img).float()
                        tmp_img = tmp_img.unsqueeze(0)
                        if index == 0:
                            imgs = tmp_img
                        else:
                            imgs = torch.cat((imgs, tmp_img), 0)
                        print('class_{}:{}, sample_name:{} ,batch_size:{}/{},seq:{}/{}'
                              .format(label.item(), data_dir_name, sample_dir_name,
                                      str(i), str(N), str(index), str(seq)))
                    print(imgs.shape)
                    dataset.append([imgs.numpy(), label.numpy()])
                    save_path = os.path.join(seq_dir, str(count) + '.npy')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    np.save(save_path, dataset)
                    count = count + 1


# ============================================================
# Augmentation helpers (NEW)
# ============================================================
def _augment_instance(img, strength='mild'):
    """
    Apply random augmentation to a single cell image.

    Args:
        img:  [H, W, C] tensor in [0, 1]
        strength: 'mild' or 'strong'
    Returns:
        augmented [H, W, C] tensor in [0, 1]
    """
    # ---- Geometric (free lunch for cell images — orientation doesn't matter) ----
    if random.random() < 0.5:
        img = torch.flip(img, dims=[1])      # horizontal flip
    if random.random() < 0.5:
        img = torch.flip(img, dims=[0])      # vertical flip

    # 90/180/270 degree rotation
    k = random.randint(0, 3)
    if k > 0:
        img = torch.rot90(img, k, dims=[0, 1])

    # ---- Photometric (mild — don't break stain colors) ----
    # Brightness
    if random.random() < 0.5:
        delta = random.uniform(-0.08, 0.08) if strength == 'mild' else random.uniform(-0.15, 0.15)
        img = (img + delta).clamp(0, 1)

    # Contrast
    if random.random() < 0.5:
        factor = random.uniform(0.9, 1.1) if strength == 'mild' else random.uniform(0.8, 1.2)
        mean = img.mean()
        img = ((img - mean) * factor + mean).clamp(0, 1)

    # Saturation (only in strong mode — risky for stain-based diagnosis)
    if strength == 'strong' and random.random() < 0.4:
        # convert to grayscale, then blend back
        gray = img.mean(dim=-1, keepdim=True).expand_as(img)
        factor = random.uniform(0.7, 1.3)
        img = (gray + factor * (img - gray)).clamp(0, 1)

    # ---- Random erasing (small patch only — don't kill the nucleus) ----
    if random.random() < 0.2:
        h, w = img.shape[0], img.shape[1]
        eh = random.randint(int(h * 0.04), int(h * 0.10))
        ew = random.randint(int(w * 0.04), int(w * 0.10))
        ey = random.randint(0, h - eh)
        ex = random.randint(0, w - ew)
        img[ey:ey + eh, ex:ex + ew, :] = random.random()

    return img


def _augment_bag(bag, strength='mild', instance_dropout_p=0.15):
    """
    Augment a whole bag.

    Args:
        bag: [N, H, W, C] tensor in [0, 1]
        strength: 'mild' or 'strong'
        instance_dropout_p: probability of dropping each instance
                            (only fires if it leaves >=8 instances).
    Returns:
        augmented bag, possibly fewer instances.
    """
    N = bag.size(0)

    # ---- Per-instance augmentation ----
    out = [_augment_instance(bag[i].clone(), strength=strength) for i in range(N)]
    bag = torch.stack(out, dim=0)

    # ---- Instance Dropout (bag-level) ----
    # Drops 0-3 instances. Only applied if >=8 instances would remain.
    if instance_dropout_p > 0 and N >= 8:
        keep_mask = torch.rand(N) > instance_dropout_p
        if keep_mask.sum() >= 8:
            bag = bag[keep_mask]

    return bag


# ============================================================
# make_data Dataset class (MODIFIED)
# ============================================================
class make_data(data.Dataset):
    def __init__(self, rootdir='dataset/', seq_dir='dataset_seq/', data_list=None,
                 seq=96, extension=1, generate=False, stage="train",
                 # ---- new args ----
                 augment=True, aug_strength='mild', instance_dropout_p=0.15):
        """
        Args (new):
            augment:         master switch for online augmentation. Default True.
            aug_strength:    'mild' (recommended start) or 'strong'.
            instance_dropout_p: probability of dropping each instance during training.
                                Set to 0 to disable. Default 0.15.

        Behavior:
            - Augmentation is ONLY applied when stage == 'train'.
            - 'val' and 'test' always return raw (un-augmented) bags.
        """
        self.rootdir = rootdir
        self.seq_dir = seq_dir
        self.seq = seq
        self.extension = extension
        self.generate = generate
        self.stage = stage                      # ★ store for use in __getitem__
        self.augment = augment
        self.aug_strength = aug_strength
        self.instance_dropout_p = instance_dropout_p

        if self.generate is False:
            print('====>loading dataset')
        else:
            print('===>Begining make dataset')
            # make_dataset(rootdir=self.rootdir, seq_dir=self.seq_dir, seq=self.seq, extension=self.extension)

        if stage == "test":
            self.data_list = glob(self.seq_dir + '/*.npy')
        else:
            self.data_list = data_list

        # Print augmentation status (helps catch silent mistakes)
        if stage == 'train' and augment:
            print(f"[dataset] train augmentation: ENABLED  "
                  f"(strength={aug_strength}, instance_dropout_p={instance_dropout_p})")
        else:
            print(f"[dataset] augmentation: disabled (stage={stage}, augment={augment})")

    def __getitem__(self, idx):
        data_path = self.data_list[idx]
        datas = np.load(data_path, allow_pickle=True)
        seq_img = torch.from_numpy(datas[0][0].astype(np.float64)).float()    # [N, H, W, C]
        seq_label = torch.tensor(datas[0][1].astype(np.float64)).long()

        # ★ Online augmentation — train only
        if self.stage == "train" and self.augment:
            seq_img = _augment_bag(
                seq_img,
                strength=self.aug_strength,
                instance_dropout_p=self.instance_dropout_p,
            )

        return data_path, seq_img, seq_label

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    rootdir = '/mnt/Disk1/lumingjie/MILMICCAI/iNHLdata'
    seq_dir = '/mnt/Disk1/lumingjie/MILMICCAI/iNHLdatanpyseq4'
    dataset = make_dataset(rootdir, seq_dir, seq=4, extension=1)