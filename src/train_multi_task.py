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

project_path = "/home/large_dataset/caozhangjie/multi-task/DRN"
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
    test_interval = config["test_interval"]
    dset_loaders = config["loaders"]
    file_out = config["file_out"]
    output_dir = config["output_dir"]

    since = time.time()
    test_iter = 1
    best_acc = 0.0
    
    len_renew = min([len(loader) - 1 for loader in dset_loaders["train"]])
    bset_acc = 0

    for iter_num in xrange(1, num_iter+1):
        if iter_num % test_interval == 0:
            epoch_acc_list = test(dset_loaders["test"], model)
            for i in xrange(num_tasks):
                print('Iter {:05d} Acc on Task {:d}: {:.4f}'.format(iter_num, i, epoch_acc_list[i]))
                file_out.write('Iter {:05d} Acc on Task {:d}: {:.4f}\n'.format(iter_num, i, epoch_acc_list[i]))
                file_out.flush()

            if np.mean(epoch_acc_list) > best_acc:
                best_acc = np.mean(epoch_acc_list)

            print('Best val Acc: {:4f}'.format(best_acc))
            file_out.write('Best val Acc: {:4f}\n'.format(best_acc))
            file_out.flush()
            save_dict = {}
            for i in xrange(len(model.networks)):
                save_dict["model"+str(i)] = model.networks[i]
            save_dict["optimizer"] = model.optimizer
            save_dict["iter_num"] = model.iter_num
            torch.save(save_dict, "../snapshot/"+output_dir+"/iter_{:05d}_model.pth.tar".format(iter_num))

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

def predict(loader, model):
    predict_list = []
    iter_ = iter(loader)
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
    return predict


def test(loaders, model):
    accuracy_list = []
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
    parser.add_argument('dset_name', type=str, nargs='?', default='0', help="dataset name")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    config = {}
    config["dset_name"] = args.dset_name
    config["gpus"] = range(len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
    config["num_tasks"] = 4
    config["num_iter"] = 30000
    config["test_interval"] = 100
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
    batch_size = {"train":24, "test":4}
    
    if config["dset_name"] == "Office":
        task_name_list = ["amazon", "webcam", "dslr", "c"]
        train_file_list = [os.path.join(project_path, "data", "office", task_name_list[i], "train_5.txt") for i in xrange(config["num_tasks"])]
        test_file_list = [os.path.join(project_path, "data", "office", task_name_list[i], "test_5.txt") for i in xrange(config["num_tasks"])]
        dset_classes = range(10)
    elif config["dset_name"] == "Office-Home":
        task_name_list = ["Art", "Product", "Clipart", "Real_World"]
        train_file_list = [os.path.join(project_path, "data", "office-home", task_name_list[i], "train_10.txt") for i in xrange(config["num_tasks"])]
        test_file_list = [os.path.join(project_path, "data", "office-home", task_name_list[i], "test_10.txt") for i in xrange(config["num_tasks"])]
        dset_classes = range(65)


    dsets = {"train": [ImageList(open(train_file_list[i]).readlines(), transform=data_transforms["train"]) for i in xrange(config["num_tasks"])], "test":[ImageList(open(test_file_list[i]).readlines(), transform=data_transforms["val9"]) for i in xrange(config["num_tasks"])]}
    dset_loaders = {"train":[], "test":[]}
    for train_dset in dsets["train"]:
        dset_loaders["train"].append(torch.utils.data.DataLoader(train_dset, batch_size=batch_size["train"], shuffle=True, num_workers=4))
    for test_dset in dsets["test"]:
        dset_loaders["test"].append(torch.utils.data.DataLoader(test_dset, batch_size=batch_size["test"], shuffle=True, num_workers=4))   
    config["loaders"] = dset_loaders


    #construct model and initialize
    config["model"] = model_multi_task.HomoMultiTaskModel(config["num_tasks"], "vgg16no_fc", len(dset_classes), config["gpus"], config["file_out"], trade_off=1, optim_param={"init_lr":0.00003, "gamma":0.3, "power":0.75, "stepsize":3000})

    #start train
    print "start train"
    config["file_out"].write("start train\n")
    config["file_out"].flush()
    experiment(config)
    config["file_out"].close()
