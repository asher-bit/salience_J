import numpy as np
import math
import os
import sys
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image

vgg16_mean = np.array([123.68, 116.779, 103.939]) / 255.
vgg16_std = np.array([0.229, 0.224, 0.225])


# tos-07-1b-000337_1 tos-02-1c-000093_0 tos-04-4a-000090 Disco_1920x1080p_30_00299 Disco_1920x1080p_30_00300 MPI-hw-630 tos-04-1b-000221
class SaliconDataset(Dataset):

    def __init__(self, dataset_root, stimuli_dir, multi_exposure_dir, fixation_maps_dir, transform=None,
                 use_cache=False, type='train'):

        super(SaliconDataset, self).__init__()

        self.dataset_root = dataset_root

        self.stimuli_dir = stimuli_dir
        self.multi_exposure_dir = multi_exposure_dir
        self.fixation_maps_dir = fixation_maps_dir

        self.image_names = [f for f in os.listdir(os.path.join(dataset_root, stimuli_dir)) if
                            os.path.isfile(os.path.join(dataset_root, stimuli_dir, f)) and not f.startswith('.')]

        self.transform = transform

        self.use_cache = use_cache

        self.lens = len(self.image_names)

    def __getitem__(self, index):
        fine_coarse_add = os.path.join(self.dataset_root, self.stimuli_dir, self.image_names[index])
        # multi-exposure image
        name_exp = self.image_names[index].split('.')[0]
        multi_exposure = os.path.join(self.dataset_root, self.multi_exposure_dir)
        # img_me Boitard_selected-Desert-1920x1080p-30-00225_crf71_a
        name_exp_1 = name_exp + '_crf71_a.png'
        img_me1 = Image.open(os.path.join(multi_exposure, name_exp_1))
        name_exp_2 = name_exp + '_crf71_b.png'
        img_me2 = Image.open(os.path.join(multi_exposure,name_exp_2))
        name_exp_3 = name_exp + '_crf71_c.png'
        img_me3 = Image.open(os.path.join(multi_exposure,name_exp_3))
        name_exp_4 = name_exp + '_crf71_d.png'
        img_me4 = Image.open(os.path.join(multi_exposure,name_exp_4))
        name_exp_5 = name_exp + '_crf71_e.png'
        img_me5 = Image.open(os.path.join(multi_exposure,name_exp_5))
        name_exp_6 = name_exp + '_crf71_f.png'
        img_me6 = Image.open(os.path.join(multi_exposure,name_exp_6))
        name_exp_7 = name_exp + '_crf71_g.png'
        img_me7 = Image.open(os.path.join(multi_exposure,name_exp_7))
        name_exp_8 = name_exp + '_crf71_h.png'
        img_me8 = Image.open(os.path.join(multi_exposure,name_exp_8))
        name_exp_9 = name_exp + '_crf71_i.png'
        img_me9 = Image.open(os.path.join(multi_exposure,name_exp_9))


        label_add = os.path.join(self.dataset_root, self.fixation_maps_dir,
                                 self.image_names[index].replace('.jpeg', '.jpg'))
        self.fine_coarse_add = fine_coarse_add

        original_img = Image.open(fine_coarse_add)
        label = Image.open(label_add).convert('L')

        if self.transform is None:
            fine_img = original_img
        elif all([x in self.transform.keys() for x in ['fine', 'label']]):
            fine_img = self.transform['fine'](original_img)
            me1_img = self.transform['fine'](img_me1)
            me2_img=self.transform['fine'](img_me2)
            me3_img=self.transform['fine'](img_me3)
            me4_img=self.transform['fine'](img_me4)
            me5_img=self.transform['fine'](img_me5)
            me6_img=self.transform['fine'](img_me6)
            me7_img=self.transform['fine'](img_me7)
            me8_img=self.transform['fine'](img_me8)
            me9_img=self.transform['fine'](img_me9)
            label = self.transform['label'](label)
        else:
            raise NotImplemented

        return [fine_img,me1_img,me2_img,me3_img, me4_img,me5_img,me6_img,me7_img,me8_img,me9_img],label,name_exp


    def __len__(self):
        return self.lens


def getTrainVal_loader(train_dataset_dir, img_dir, multi_exposure_dir, label_dir, shuffle=True, val_split=0.1):
    vgg16_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32) / 255.
    vgg16_std = np.array([0.229, 0.224, 0.225])

    data_transforms = {
        'fine': transforms.Compose([
            transforms.Resize((600, 800), interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ]),
        'multi_exposure': transforms.Compose([
            transforms.Resize((400, 600), interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ]),
        'label': transforms.Compose([
            transforms.Resize((600, 800), interpolation=0),
            transforms.ToTensor(),

        ])
    }

    trainval_dataset = SaliconDataset(train_dataset_dir, img_dir, multi_exposure_dir, label_dir,
                                      transform=data_transforms)

    dataset_size = len(trainval_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    if shuffle:
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(trainval_dataset, batch_size=1, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(trainval_dataset, batch_size=1, sampler=valid_sampler, num_workers=0)

    trainval_loaders = {'train': train_loader, 'val': val_loader}

    return trainval_loaders


def getTest_loader(test_dataset_dir, img_dir, multi_exposure_dir, label_dir, shuffle=True):
    vgg16_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32) / 255.
    vgg16_std = np.array([0.229, 0.224, 0.225])

    data_transforms = {
        'fine': transforms.Compose([
            transforms.Resize((600, 800), interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ]),
        'coarse': transforms.Compose([
            transforms.Resize((300, 400), interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ]),
        'label': transforms.Compose([
            # transforms.Resize((37, 50), interpolation=2),
            transforms.ToTensor(),
            # transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ])
    }

    test_dataset = SaliconDataset(test_dataset_dir, img_dir, multi_exposure_dir, label_dir, transform=data_transforms)

    test_loader=DataLoader(test_dataset,batch_size=1,num_workers=6)

    return test_loader


if __name__ == '__main__':
    project_dir = os.path.abspath('..')
    # print(project_dir)
    mit1003_dir = project_dir + '/mit1003_dataset'

    osie_dir = project_dir + '/osie_dataset/data'

    data_transforms = {
        'fine': transforms.Compose([
            transforms.Resize((600, 800), interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ]),
        'coarse': transforms.Compose([
            transforms.Resize((300, 400), interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ]),
        'label': transforms.Compose([
            transforms.Resize((152, 204), interpolation=0),
            transforms.ToTensor(),
            # transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ])
    }

    test_dataset = SaliconDataset(mit1003_dir, 'ALLSTIMULI', 'ALLFIXATIONMAPS', transform=data_transforms)

    trainval_dataset = SaliconDataset(osie_dir, 'stimuli', 'fixation_maps', transform=data_transforms)

    # divide trainval dataset to train dataset and val dataset
    shuffle_dataset = True
    val_split = 0.1
    random_seed = 42

    dataset_size = len(trainval_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    loader_train = DataLoader(trainval_dataset, batch_size=1, sampler=train_sampler, num_workers=1)
    loader_val = DataLoader(trainval_dataset, batch_size=1, sampler=valid_sampler, num_workers=1)

    input, output = (next(iter(loader_train)))
    fine_img, coarse_img = input
    print(fine_img.shape, coarse_img.shape)
