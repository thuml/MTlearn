# MRN
Code for "Learning Multiple Tasks with Multilinear Relationship Networks" (NIPS 2017)

## Prerequisites
Linux or OSX

NVIDIA GPU + CUDA-7.5 or CUDA-8.0 and corresponding CuDNN

PyTorch

Python 2.7

## Datasets
### Office-Caltech
We use the shared classes of [Office-31](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/) dataset and [Caltech-256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/) dataset to construct a multi-task dataset of four tasks, which are Amazon, Webcam, Dslr, Caltech. The corresponding list files are [here](./data/office). In each sub-directory, there are three training list files called train_5.txt, train_10.txt and train_20.txt and three test list files called test_5.txt, test_10.txt, test_20.txt which are corresponding to the training and test files of 5% data, 10% data and 20% data.

### Office-Home
We use [Office-Home](http://hemanthdv.org/OfficeHome-Dataset/) dataset which composes of 65 classes to construct a multi-task dataset of four tasks, which are Art, Clipart, Product, Real World. The corresponding list files are [here](./data/office-home). In each sub-directory, there are three training list files called train_5.txt, train_10.txt and train_20.txt and three test list files called test_5.txt, test_10.txt, test_20.txt which are corresponding to the training and test files of 5% data, 10% data and 20% data.

## Training
Use the following command to run the training code.
```
cd src 

dataset_name = Office or Office-Home
python train_multi_task.py gpu_id dataset_name
```

Whether to finetune the pre-trained CNN layers depends on the size of your training dataset. In our experiment, since 5%, 10%, 20% of Office-Caltech dataset and 5% of Office-Home dataset has too few data, we do not finetune the CNN layers to get rid of overfitting. For 10% and 20% of Office-Home dataset, we finetune the whole network. You can use the base CNN model to test whether your dataset need finetuning or not.

Change "lr" in code 
```
parameter_dict = [{"params":self.shared_layers.module.parameters(), "lr":0}]
```
in the 'model_multi_task.py' file to control whether to finetune the base CNN layers. 0 is not finetuning.

## Evaluation
The evaluation results are showed in the training log and are also printed in the training process.

You can also use the 'predict' function in the code to predict your result of images.


## Citation
If you use this code for your research, please consider citing:
```
@incollection{NIPS2017_6757,
title = {Learning Multiple Tasks with Multilinear Relationship Networks},
author = {Long, Mingsheng and CAO, ZHANGJIE and Wang, Jianmin and Yu, Philip S},
booktitle = {Advances in Neural Information Processing Systems 30},
editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
pages = {1593--1602},
year = {2017},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/6757-learning-multiple-tasks-with-multilinear-relationship-networks.pdf}
}
```
## Contact
If you have any problem about our code, feel free to contact caozhangjie14@gmail.com or describe your problem in Issues.
