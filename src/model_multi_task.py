import torch
import torch.nn as nn
import torchvision
from torchvision import models
import tensor_op
from torch.autograd import Variable
import torch.optim as optim

# convnet without the last layer
class AlexnetFc(nn.Module):
  def __init__(self, output_num):
    super(AlexnetFc, self).__init__()
    model_alexnet = models.alexnet(pretrained=True)
    self.features = model_alexnet.features
    self.classifier = nn.Sequential()
    for i in xrange(6):
      self.classifier.add_module("classifier"+str(i), model_alexnet.classifier[i])
    self.__in_features = model_alexnet.classifier[6].in_features
    self.output_num = output_num
    self.bottleneck = nn.Linear(self.__in_features, 256)
    self.fc = nn.Linear(256, output_num)
    self.bottleneck.weight.data.normal_(0, 0.005)
    self.fc.weight.data.normal_(0, 0.01)
    self.bottleneck.bias.data.fill_(0.1)
    self.fc.bias.data.fill_(0.0)

  
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), 256*6*6)
    x = self.classifier(x)
    x = self.bottleneck(x)
    x = self.fc(x)
    return x

  def output_num(self):
    return self.output_num

class Resnet18Fc(nn.Module):
  def __init__(self, output_num):
    super(Resnet18Fc, self).__init__()
    model_resnet18 = models.resnet18(pretrained=True)
    self.conv1 = model_resnet18.conv1
    self.bn1 = model_resnet18.bn1
    self.relu = model_resnet18.relu
    self.maxpool = model_resnet18.maxpool
    self.layer1 = model_resnet18.layer1
    self.layer2 = model_resnet18.layer2
    self.layer3 = model_resnet18.layer3
    self.layer4 = model_resnet18.layer4
    self.avgpool = model_resnet18.avgpool
    self.__in_features = model_resnet18.fc.in_features
    self.output_num = output_num
    self.bottleneck = nn.Linear(self.__in_features, 256)
    self.fc = nn.Linear(256, output_num)
    self.bottleneck.weight.data.normal_(0, 0.005)
    self.fc.weight.data.normal_(0, 0.01)
    self.bottleneck.bias.data.fill_(0.1)
    self.fc.bias.data.fill_(0.0)


  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.bottleneck(x)
    x = self.fc(x)
    return x

  def output_num(self):
    return self.output_num

class Resnet34Fc(nn.Module):
  def __init__(self, output_num):
    super(Resnet34Fc, self).__init__()
    model_resnet34 = models.resnet34(pretrained=True)
    self.conv1 = model_resnet34.conv1
    self.bn1 = model_resnet34.bn1
    self.relu = model_resnet34.relu
    self.maxpool = model_resnet34.maxpool
    self.layer1 = model_resnet34.layer1
    self.layer2 = model_resnet34.layer2
    self.layer3 = model_resnet34.layer3
    self.layer4 = model_resnet34.layer4
    self.avgpool = model_resnet34.avgpool
    self.__in_features = model_resnet34.fc.in_features
    self.output_num = output_num
    self.bottleneck = nn.Linear(self.__in_features, 256)
    self.fc = nn.Linear(256, output_num)
    self.bottleneck.weight.data.normal_(0, 0.005)
    self.fc.weight.data.normal_(0, 0.01)
    self.bottleneck.bias.data.fill_(0.1)
    self.fc.bias.data.fill_(0.0)


  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.bottleneck(x)
    x = self.fc(x)
    return x

  def output_num(self):
    return self.output_num

class Resnet50Fc(nn.Module):
  def __init__(self, output_num):
    super(Resnet50Fc, self).__init__()
    model_resnet50 = models.resnet50(pretrained=True)
    self.conv1 = model_resnet50.conv1
    self.bn1 = model_resnet50.bn1
    self.relu = model_resnet50.relu
    self.maxpool = model_resnet50.maxpool
    self.layer1 = model_resnet50.layer1
    self.layer2 = model_resnet50.layer2
    self.layer3 = model_resnet50.layer3
    self.layer4 = model_resnet50.layer4
    self.avgpool = model_resnet50.avgpool
    self.__in_features = model_resnet50.fc.in_features
    self.output_num = output_num
    self.bottleneck = nn.Linear(self.__in_features, 256)
    self.fc = nn.Linear(256, output_num)
    self.bottleneck.weight.data.normal_(0, 0.005)
    self.fc.weight.data.normal_(0, 0.01)
    self.bottleneck.bias.data.fill_(0.1)
    self.fc.bias.data.fill_(0.0)


  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.bottleneck(x)
    x = self.fc(x)
    return x

  def output_num(self):
    return self.output_num

class Resnet101Fc(nn.Module):
  def __init__(self, output_num):
    super(Resnet101Fc, self).__init__()
    model_resnet101 = models.resnet101(pretrained=True)
    self.conv1 = model_resnet101.conv1
    self.bn1 = model_resnet101.bn1
    self.relu = model_resnet101.relu
    self.maxpool = model_resnet101.maxpool
    self.layer1 = model_resnet101.layer1
    self.layer2 = model_resnet101.layer2
    self.layer3 = model_resnet101.layer3
    self.layer4 = model_resnet101.layer4
    self.avgpool = model_resnet101.avgpool
    self.extract_feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

    self.in_features = model_resnet101.fc.in_features
    self.output_num = output_num
    self.bottleneck = nn.Linear(self.in_features, 256)
    self.fc = nn.Linear(256, output_num)
    self.bottleneck.weight.data.normal_(0, 0.005)
    self.fc.weight.data.normal_(0, 0.01)
    self.bottleneck.bias.data.fill_(0.1)
    self.fc.bias.data.fill_(0.0)


  def forward(self, x):
    x = self.extract_feature_layers(x)
    x = x.view(x.size(0), -1)
    x = self.bottleneck(x)
    x = self.fc(x)
    return x

  def output_num(self):
    return self.output_num


class Resnet152Fc(nn.Module):
  def __init__(self, output_num):
    super(Resnet152Fc, self).__init__()
    model_resnet152 = models.resnet152(pretrained=True)
    self.conv1 = model_resnet152.conv1
    self.bn1 = model_resnet152.bn1
    self.relu = model_resnet152.relu
    self.maxpool = model_resnet152.maxpool
    self.layer1 = model_resnet152.layer1
    self.layer2 = model_resnet152.layer2
    self.layer3 = model_resnet152.layer3
    self.layer4 = model_resnet152.layer4
    self.avgpool = model_resnet152.avgpool
    self.__in_features = model_resnet152.fc.in_features
    self.output_num = output_num
    self.bottleneck = nn.Linear(self.__in_features, 256)
    self.fc = nn.Linear(256, output_num)
    self.bottleneck.weight.data.normal_(0, 0.005)
    self.fc.weight.data.normal_(0, 0.01)
    self.bottleneck.bias.data.fill_(0.1)
    self.fc.bias.data.fill_(0.0)


  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.bottleneck(x)
    x = self.fc(x)

    return x

  def output_num(self):
    return self.output_num


network_dict = {"resnet101":Resnet101Fc}

class HomoMultiTaskModel(object):
    def __init__(self, num_tasks, network_list, output_num, gpus, optim_param={"init_lr":0.0003, "gamma":0.001, "power":0.75}, cov_fc7=True):
        self.train_cross_loss = 0
        self.train_multi_task_loss = 0
        self.train_total_loss = 0
        self.print_interval = 500

        self.num_tasks = num_tasks
        self.network_list = network_list
        self.output_num = output_num
        self.bottleneck_size = 256
        self.num_gpus = len(gpus)
        self.networks = [network_dict[self.network_list[i]](self.output_num).cuda() for i in xrange(num_tasks)]
        self.networks = [nn.DataParallel(network, device_ids=gpus) for network in self.networks]
        parameter_dict = [{"params":self.networks[i].module.extract_feature_layers.parameters(), "lr":1} for i in xrange(self.num_tasks)]
        parameter_dict += [{"params":self.networks[i].module.bottleneck.parameters(), "lr":10} for i in xrange(self.num_tasks)]
        parameter_dict += [{"params":self.networks[i].module.fc.parameters(), "lr":10} for i in xrange(self.num_tasks)]
        self.optimizer = optim.SGD(parameter_dict, lr=1, momentum=0.9, weight_decay=0.0005)
        self.parameter_lr_dict = []
        for param_group in self.optimizer.param_groups:
            self.parameter_lr_dict.append(param_group["lr"])
        self.optim_param = {"init_lr":0.0003, "gamma":0.001, "power":0.75}
        for val in optim_param:
            self.optim_param[val] = optim_param[val]

        self.task_cov_fc8 = torch.eye(num_tasks)
        self.class_cov_fc8 = torch.eye(output_num)
        self.feature_cov_fc8 = torch.eye(self.bottleneck_size)
        self.task_cov_fc8_v = Variable(self.task_cov_fc8).cuda()
        self.class_cov_fc8_v = Variable(self.class_cov_fc8).cuda()
        self.feature_cov_fc8_v = Variable(self.feature_cov_fc8).cuda()
        if cov_fc7:
            self.task_cov_fc7 = torch.eye(num_tasks)
            self.class_cov_fc7 = torch.eye(self.networks.bottleneck_size)
            self.feature_cov_fc7 = torch.eye(self.in_features)
            self.task_cov_fc7_v = Variable(self.task_cov_fc7).cuda()
            self.class_cov_fc7_v = Variable(self.class_cov_fc7).cuda()
            self.feature_cov_fc7_v = Variable(self.feature_cov_fc7).cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.iter_num = 1
       

    def optimize_model(self, input_list, label_list):
        # update learning rate
        current_lr = self.optim_param["init_lr"] * ((1 + self.optim_param["gamma"] * self.iter_num) ** self.optim_param["power"])
        i = 0
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr * self.parameter_lr_dict[i]
            i += 1

        # losses 
        for i in xrange(self.num_tasks):
            self.networks[i].train(True)
        self.optimizer.zero_grad()
        output_list = [self.networks[i](input_list[i]) for i in xrange(self.num_tasks)]
        losses = [self.criterion(output_list[i], label_list[i]) for i in xrange(self.num_tasks)]
        classifier_loss = sum(losses)
        weight_size = self.networks[0].module.fc.weight.size()
        all_weights = [self.networks[i].module.fc.weight.view(1, weight_size[0], weight_size[1]) for i in xrange(self.num_tasks)]
        weights_fc8 = torch.cat(all_weights, dim=0).contiguous() 
        weight_size = self.networks[0].module.bottleneck.weight.size()
        all_weights = [self.networks[i].module.bottleneck.weight.view(1, weight_size[0], weight_size[1]) for i in xrange(self.num_tasks)]
        weights_fc7 = torch.cat(all_weights, dim=0).contiguous() 
        multi_task_loss = tensor_op.MultiTaskLoss(weights_fc8, self.task_cov_fc8_v, self.class_cov_fc8_v, self.feature_cov_fc8_v)
        total_loss = classifier_loss + multi_task_loss
        # update network parameters
        total_loss.backward()
        self.optimizer.step()

        # get updated weights
        weight_size = self.networks[0].module.fc.weight.size()
        all_weights = [self.networks[i].module.fc.weight.view(1, weight_size[0], weight_size[1]) for i in xrange(self.num_tasks)]
        weights_fc8 = torch.cat(all_weights, dim=0).contiguous() 
        weight_size = self.networks[0].module.bottleneck.weight.size()
        all_weights = [self.networks[i].module.bottleneck.weight.view(1, weight_size[0], weight_size[1]) for i in xrange(self.num_tasks)]
        weights_fc7 = torch.cat(all_weights, dim=0).contiguous() 

        # update cov parameters
        temp_task_cov_fc8_v = tensor_op.UpdateCov(weights_fc8.data, self.class_cov_fc8_v.data, self.feature_cov_fc8_v.data)
        temp_class_cov_fc8_v = tensor_op.UpdateCov(weights_fc8.data.permute(1, 0, 2).contiguous(), self.task_cov_fc8_v.data, self.feature_cov_fc8_v.data)
        temp_feature_cov_fc8_v = tensor_op.UpdateCov(weights_fc8.data.permute(2, 0, 1).contiguous(), self.task_cov_fc8_v.data, self.class_cov_fc8_v.data)
        self.task_cov_fc8_v = Variable(temp_task_cov_fc8_v / torch.trace(temp_task_cov_fc8_v)).cuda()
        self.class_cov_fc8_v = Variable(temp_class_cov_fc8_v / torch.trace(temp_class_cov_fc8_v)).cuda()
        self.feature_cov_fc8_v = Variable(temp_feature_cov_fc8_v / torch.trace(temp_feature_cov_fc8_v)).cuda()

        self.iter_num += 1
        if self.iter_num % self.print_interval == 0:
            print("Iter {:05d}, Average Cross Entropy Loss: {:.4f}; Average MultiTask Loss: {:.4f}; Average Training Loss: {:.4f}".format(self.iter_num, self.train_cross_loss / float(self.print_interval), self.train_multi_task_loss / float(self.print_interval), self.train_total_loss / float(self.print_interval)))
            self.train_cross_loss = 0
            self.train_multi_task_loss = 0
            self.train_total_loss = 0

       
    def test_model(self, input_, i):
        self.networks[i].train(False)
        output = self.networks[i](input_)
        return output 
