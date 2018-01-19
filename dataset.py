import torch.utils.data as data

from PIL import Image
import os
import re
import torch
import pdb
import random
import pickle
import urllib.request as request
# import attacker
from torchvision import transforms

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


# ImageId,URL,x1,y1,x2,y2,TrueLabel,TargetClass,OriginalLandingURL,License,Author,AuthorProfileURL
def construct_class_to_idx(root):
    file_path = os.path.join(root, "images.csv")
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.split(',') for line in lines][1:]
    img_targets = dict()
    img_true = dict()
    for l in lines:
        # pdb.set_trace()
        img_targets[l[0]] = l[7]
        img_true[l[0]] = l[6]
    return img_targets, img_true


def find_images_and_targets(folder, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True):
    if class_to_idx is None:
        class_to_idx = dict()
        build_class_idx = True
    else:
        build_class_idx = False
    labels = []
    filenames = []

    for root, subdirs, files in os.walk(folder, topdown=False):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        # pdb.set_trace()
        if build_class_idx and not subdirs:
            class_to_idx[label] = 0
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root,f))
                labels.append(label)
    if build_class_idx:
        classes = sorted(class_to_idx.keys(), key=natural_key)
        for idx, c in enumerate(classes):
            class_to_idx[c] = idx

    images_and_targets = zip(filenames, [class_to_idx[l] for l in labels])
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    if build_class_idx:
        return images_and_targets, classes, class_to_idx
    else:
        return images_and_targets


def find_class_targets_and_true(data_root, img_targets, img_true_labels, types=IMG_EXTENSIONS):
    filenames = []
    targets = []
    labels = []
    for f in os.listdir(data_root):
        base, ext = os.path.splitext(f)
        if ext.lower() in types:
            # pdb.set_trace()
            filenames.append(os.path.join(data_root, f))
            targets.append(img_targets[base])
            labels.append(img_true_labels[base])
    # pdb.set_trace()
    return list(zip(filenames, targets)), list(zip(filenames, labels))


def get_classes(root):
    file_path = os.path.join(root, "categories.csv")
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]
        classes = [l[1] for l in lines]
    return classes


def create_target(img_and_label, n_classes):
    # filenames = [img[0] for img in imgs_and_label]
    imgs_and_targets = []
    for f, label in img_and_label:
        # pdb.set_trace()
        while True:
            rd = random.randint(0, n_classes - 1)
            if rd == label:
                continue
            break
        imgs_and_targets.append((f, rd))

    return imgs_and_targets




class Dataset(data.Dataset):

    def __init__(self, root, transform=None):
        # imgs, classes, class_to_idx = find_images_and_targets(root)
        img_targets, img_true = construct_class_to_idx(root)
        data_path = os.path.join(root, "images/")
        self.img_and_targets, self.img_and_labels = find_class_targets_and_true(data_path, img_targets, img_true)

        # specify for train, test and val   --->  root
        imgs, classes, class_to_idx = find_images_and_targets(root)


        # imgs = find_images_and_targets(root, class_to_idx=img_targets)
        if len(self.img_and_targets) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        class_num = 1000
        self.root = root
        # self.imgs = imgs
        self.classes = classes
        self.n_classes = 1000
        # self.class_to_idx = self.img_targets
        self.transform = transform

    def __getitem__(self, index):
        # pdb.set_trace()
        path, target = self.img_and_targets[index]
        _, true = self.img_and_labels[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)


        # Use for un-targeted attack!
        # if target is None:
        #     target = torch.zeros(1).long()
        # else:
            # target is str
            # target = int(target)
        target = torch.LongTensor([int(target),])
        true = torch.LongTensor([int(true), ])
        # pdb.set_trace()
        return img, target, true

    def __len__(self):
        return len(self.img_and_targets)

    def filenames(self, indices=[]):
        if indices:
            return [self.img_and_targets[i][0] for i in indices]
        else:
            return [x[0] for x in self.img_and_targets]






class imageNet(data.Dataset):
    def __init__(self, root, transform=None):
        # preprocess ---> got label dicts
        self.class_to_ix, self.ix_to_l = self.preprocess()
        self.n_classes = 1000
        self.transform = transform
        # a,b,c = find_images_and_targets(root)
        self.img_and_labels = self.find_img_and_label(root, self.class_to_ix)
        self.img_and_targets = create_target(self.img_and_labels, self.n_classes)
        # pdb.set_trace()

        return

    def __getitem__(self, index):
        path, target = self.img_and_targets[index]
        # print(path)
        _, true = self.img_and_labels[index]
        img = Image.open(path).resize((299, 299), resample=Image.LANCZOS)

        if self.transform is not None:
            img = self.transform(img)

        target = torch.LongTensor([int(target),])
        true = torch.LongTensor([int(true), ])
        # pdb.set_trace()
        return img, target, true

    def __len__(self):
        return len(self.img_and_labels)


    def find_img_and_label(self, folder, l_to_ix, types=IMG_EXTENSIONS):
        filenames = []
        labels = []
        for root, sub, files in os.walk(folder, topdown=False):
            # print(root)
            for f in files:
                tmp = os.path.split(root)[-1]
                if len(tmp) < 3:
                    continue
                label = tmp
                # print(os.path.join(root, f))
                # filenames.append(os.path.join(root, f))
                # print(len(labels))
                labels.append(l_to_ix[label])
                filenames.append(os.path.join(root, f))
        return list(zip(filenames, labels))


    def preprocess(self):
        file_path = "/home/image.txt"
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # n02...  --- > ice-cream
        class_to_l = {}
        for l in lines:
            a, b = l.strip().split(":")
            class_to_l[a.strip()] = b.strip()

        l_to_ix = dict()

        # tmp is ix -> l
        # 924: 'guacamole'
        tmp = pickle.load(request.urlopen('https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'))
        for key, item in tmp.items():
            l_to_ix[item] = key

        class_to_ix = {}
        for key, item in class_to_l.items():
            class_to_ix[key] = l_to_ix[item]

        return class_to_ix, tmp


    def check(self):
        # pdb.set_trace()
        wrong_img = []
        for path, target in self.img_and_targets:
            img = Image.open(path).resize((299, 299), resample=Image.LANCZOS)
            if self.transform is not None:
                img = self.transform(img)
            # print(img.shape)
            if img.shape[0] == 1:
                print(path)
                # wrong_img.append(path)
                cmd = "rm -f %s"%path
                # print(cmd)
                os.system(cmd)
        return wrong_img


if __name__ == "__main__":
    root = "/home/Data/"
    img_size = 299

    tf = transforms.Compose([
        #     transforms.Scale(int(math.floor(img_size / 0.875))),
        transforms.CenterCrop(img_size),
        # normalize,
        transforms.ToTensor(),
        # LeNormalize(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    w_imgs = imageNet(root, transform=tf).check()


