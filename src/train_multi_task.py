#from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, transforms
#import matplotlib.pyplot as plt
import time
import copy
import os
from PIL import Image, ImageOps
import numbers
import argparse
import pickle
import random
import sys

project_path = "/home/large_dataset/caozhangjie/multi-task/pytorch"
#sys.path.append(os.path.join("/home/caozhangjie/run-czj/wgan-pytorch/alexnet", "src"))

import model_multi_task
import caffe_transform as caffe_t
from data import ImageList


def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (-power)

    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i+=1

    return optimizer

# train function
def experiment(config):
    model = config["model"]
    num_iter = config["num_iter"]
    num_tasks = config["num_tasks"]
    #test_interval = config["test_interval"]
    whole_test_interval = config["whole_test_interval"]
    dset_loaders = config["loaders"]
    file_out = config["file_out"]
    output_dir = config["output_dir"]

    since = time.time()
    test_iter = 1
    best_acc = 0.0
    
    len_renew = min([len(loader) - 1 for loader in dset_loaders["train"]])
    bset_acc = 0

    for iter_num in xrange(1, num_iter+1):
        if iter_num % whole_test_interval == 1:
            epoch_acc_list = test_target(dset_loaders["test"], model)
            for i in xrange(num_tasks):
                print('Iter {:05d} Acc on Task {:d}: {:.4f}\n'.format(iter_num, i, epoch_acc_list[i]))  

            if np.mean(epoch_acc_list) > best_acc:
                best_acc = np.mean(epoch_acc_list)

            print('Best val Acc: {:4f}'.format(best_acc))
            save_dict = {}
            for i in xrange(len(model.networks)):
                save_dict["model"+str(i)] = model.networks[i]
            save_dict["optimizer"] = model.optimizer
            save_dict["iter_num"] = model.iter_num
            torch.save(save_dict, "../snapshot/"+output_dir+"/iter_{:05d}_model.pth.tar".format(iter_num))

        if iter_num % 500 == 0:
            print("Iter {:05d}".format(iter_num))

        if (iter_num-1) % len_renew == 0:
            iter_list = [iter(loader) for loader in dset_loaders["train"]]
        data_list = []
        for iter_ in iter_list:
            data_list.append(iter_.next())
        # get the inputs
        input_list = []
        label_list = []
        for one_data in data_list:
            inputs, labels = one_data
            input_list.append(Variable(inputs).cuda())
            label_list.append(Variable(labels).cuda())
        model.optimize_model(input_list, label_list)

        #if iter_num % test_interval == 0:
        #    epoch_acc = test_target(dset_loaders, model_list["predict"], test_iter=config["test_iter"])
        #    print('Iter {:05d} Acc on the Sample: {:.4f}'.format(iter_num, epoch_acc))
        #    file_out.write('Iter {:05d} Acc on the Sample: {:.4f}\n'.format(iter_num, epoch_acc))
        #    file_out.flush()

def predict_test(loader, model):
    start_test = True
    iter_val = iter(loader["test"])
    for i in xrange(len(loader['test'])):
        data = iter_val.next()
        inputs = data[0]
        labels = data[1]
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        outputs = model(inputs)
        if start_test:
            all_output = outputs.data.cpu().float()
            start_test = False
        else:
            all_output = torch.cat((all_output, outputs.data.cpu().float()), 0)
    _, predict = torch.max(all_output, 1)
    return all_output, predict


def test_target(loaders, model, test_iter=0):
    accuracy_list = []
    if test_iter > 0:
        pass
    else:
        iter_val = [iter(loader) for loader in loaders]
        for i in xrange(len(iter_val)):
            iter_ = iter_val[i]
            start_test = True
            for j in xrange(len(loaders[i])):
                data = iter_.next()
                inputs, labels = data
                inputs= Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                outputs = model.test_model(inputs, i)
                if start_test:
                    all_output = outputs.data.float()
                    all_label = labels.data.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.data.float()), 0)
                    all_label = torch.cat((all_label, labels.data.float()), 0)
            _, predict = torch.max(all_output, 1)
            accuracy_list.append(torch.sum(torch.squeeze(predict).float() == all_label) / float(all_label.size()[0]))
        return accuracy_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('gpu_id', type=str, nargs='?', default='0', help="device id to run")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"

    config = {}
    config["gpus"] = [0, 1, 2, 3]
    config["num_tasks"] = 4
    config["num_iter"] = 100000
    #config["test_interval"] = 1000
    #config["test_iter"] = 500
    config["whole_test_interval"] = 500
    config["output_dir"] = "pytorch_multi_task"
    os.system("mkdir -p ../snapshot/"+ config["output_dir"])
    config["file_out"] = open("../snapshot/"+ config["output_dir"]+"/train_log.txt", "w")

    #set data_transforms
    data_transforms = {
        'train': caffe_t.transform_train(resize_size=256, crop_size=224),
        'val': caffe_t.transform_train(resize_size=256, crop_size=224),
    }
    data_transforms = caffe_t.transform_test(data_transforms=data_transforms, resize_size=256, crop_size=224)
  
    #set dataset
    batch_size = {"train":30, "test":4}
    '''
    source_dir = "/home/caozhangjie/caffe-grl/data/office/domain_adaptation_images/amazon/images/"
    target_dir = "/home/caozhangjie/caffe-grl/data/office/domain_adaptation_images/webcam/images/"
    data_dir = {}
    data_dir["train"] = source_dir
    data_dir["val"] = target_dir
    dsets = {x: datasets.ImageFolder(data_dir["train"] if "train" in x else data_dir["val"], data_transforms[x])
        for x in ['train',"val"]+["val"+str(i) for i in xrange(10)]}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size[x],
                              shuffle=True if x=="train" or x=="val" else False, num_workers=4)
                  for x in ['train','val']+["val"+str(i) for i in xrange(10)]}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']+["val"+str(i) for i in xrange(10)]}
    dset_classes = dsets["train"].classes
    '''
    task_name_list = ["amazon", "webcam", "dslr", "c"]
    train_file_list = [os.path.join(project_path, "data", "office", task_name_list[i], "train_20.txt") for i in xrange(config["num_tasks"])]
    test_file_list = [os.path.join(project_path, "data", "office", task_name_list[i], "test_20.txt") for i in xrange(config["num_tasks"])]
    dsets = {"train": [ImageList(open(train_file_list[i]).readlines(), transform=data_transforms["train"]) for i in xrange(config["num_tasks"])], "test":[ImageList(open(test_file_list[i]).readlines(), transform=data_transforms["val9"]) for i in xrange(config["num_tasks"])]}

    dset_loaders = {"train":[], "test":[]}
    for train_dset in dsets["train"]:
        dset_loaders["train"].append(torch.utils.data.DataLoader(train_dset, batch_size=batch_size["train"], shuffle=True, num_workers=4))
    for test_dset in dsets["test"]:
        dset_loaders["test"].append(torch.utils.data.DataLoader(test_dset, batch_size=batch_size["test"], shuffle=True, num_workers=4))
    dset_classes = range(12)

    
    config["loaders"] = dset_loaders


    #construct model and initialize
    use_gpu = torch.cuda.is_available()    
    network_list = ["resnet101" for i in xrange(config["num_tasks"])]
    config["model"] = model_multi_task.HomoMultiTaskModel(config["num_tasks"], network_list, len(dset_classes), config["gpus"], optim_param={"init_lr":0.0003, "gamma":0.001, "power":0.75}, cov_fc7=False)
  
    #start train
    print "start train"
    experiment(config)
    config["file_out"].close()
