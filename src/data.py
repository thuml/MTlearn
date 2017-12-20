#from __future__ import print_function, division

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from PIL import Image
import torch.utils.data as data
import os
import os.path
import scipy.io as ioc

def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in xrange(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    img = Image.open(path)
    return img.convert('RGB')    


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    #from torchvision import get_image_backend
    #if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    #else:
        return pil_loader(path)


class OmniglotList(object):
    def __init__(self, mat_file_name, percent):
        mat_file = ioc.loadmat(mat_file_name)
        images = mat_file["images"]
        num_tasks = mat_file["images"].shape[0]
        image_tasks = [mat_file["images"][i, 0] for i in xrange(num_tasks)]
        self.train_images = []
        self.train_labels = []
        self.test_images = []
        self.test_labels = []
        self.class_nums = []
        for i in xrange(num_tasks):
            train_images_for_one = []
            train_labels_for_one = []
            test_images_for_one = []
            test_labels_for_one = []
            num_labels = image_tasks[i].shape[0]
            self.class_nums.append(num_labels)
            for j in xrange(num_labels):
                images_for_label = image_tasks[i][j, 0]
                num_images = images_for_label.shape[0]
                train_id = random.sample(range(num_images), int(num_images*percent/100))
                for k in xrange(num_images):                   
                    if k not in train_id:
                        size_image = images_for_label[k, 0].shape
                        test_images_for_one.append(np.reshape(images_for_label[k, 0], (1, size_image[0], size_image[1])))
                        test_labels_for_one.append(j)
                    else:
                        size_image = images_for_label[k, 0].shape
                        train_images_for_one.append(np.reshape(images_for_label[k, 0], (1, size_image[0], size_image[1])))
                        train_labels_for_one.append(j)

            self.train_images.append(np.array(train_images_for_one))
            self.train_labels.append(np.array(train_labels_for_one))
            self.test_images.append(np.array(test_images_for_one))
            self.test_labels.append(np.array(test_labels_for_one))


    def get_set(self):
        return self.train_images, self.train_labels, self.test_images, self.test_labels, self.class_nums

class OmniglotTask(object):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels)
        self.image_num = self.images.size(0)
        num0 = self.image_num
        while num0 < 32:
            num0 += self.image_num 
        self.true_num = self.image_num
        self.image_num = num0

    def __getitem__(self, index):
        return self.images[index%self.true_num, :, :, :], self.labels[index%self.true_num]  

    def __len__(self):
        return self.image_num      

class ImageList(object):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
