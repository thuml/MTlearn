import torch
import torch.nn as nn
import torchvision
from torchvision import models
import tensor_op
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

# classification model without the last layer
class Vgg16NoFc(nn.Module):
    def __init__(self):
        super(Vgg16NoFc, self).__init__()
        model_vgg = models.vgg16(pretrained=True)
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in xrange(6):
            self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
        self.extract_feature_layers = nn.Sequential(self.features, self.classifier)
        self.in_features = model_vgg.classifier[6].in_features
  
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self.in_features

network_dict = {"vgg16no_fc": Vgg16NoFc}

class HomoMultiTaskModel(object):
    # num_tasks: number of tasks
    # network_name: the base model used, add new network name in the above 'network_dict'
    # output_num: the output dimension of all the tasks
    # gpus: gpu id used (list)
    # file_out: log file
    # trade_off: the trade_off between multitask loss and task loss
    # optim_param: optimizer parameters
    def __init__(self, num_tasks, network_name, output_num, gpus, file_out, trade_off=1.0, optim_param={"init_lr":0.00003, "gamma":0.3, "power":0.75, "stepsize":3000}):
        def select_func(x):
            if x > 0.1:
                return 1. / x
            else:
                return x

        self.file_out = file_out
        # threshold function in filtering too small singular value 
        self.select_func = select_func

        self.trade_off = trade_off
        self.train_cross_loss = 0
        self.train_multi_task_loss = 0
        self.train_total_loss = 0
        self.print_interval = 500

        # covariance update frequency (one every #param iter)
        self.cov_update_freq = 100

        # construct multitask model with shared part and related part
        self.num_tasks = num_tasks
        self.network_name = network_name
        self.output_num = output_num       
        self.num_gpus = len(gpus)
        self.shared_layers = network_dict[self.network_name]().cuda() # layers shared
        self.networks = [[nn.Linear(self.shared_layers.output_num(), self.output_num).cuda()] for i in xrange(self.num_tasks)] # layers not shared but related
        for i in xrange(self.num_tasks):
            for layer in self.networks[i]:
                layer.weight.data.normal_(0, 0.01)
                layer.bias.data.fill_(0.0)
        self.networks = [nn.Sequential(*val) for val in self.networks]

        self.bottleneck_size = self.networks[0][-1].in_features

        self.shared_layers = nn.DataParallel(self.shared_layers, device_ids=gpus)
        self.networks = [nn.DataParallel(network, device_ids=gpus) for network in self.networks]

        # construct optimizer
        parameter_dict = [{"params":self.shared_layers.module.parameters(), "lr":0}]
        parameter_dict += [{"params":self.networks[i].module.parameters(), "lr":10} for i in xrange(self.num_tasks)]
        self.optimizer = optim.SGD(parameter_dict, lr=1, momentum=0.9, weight_decay=0.0005)
        self.parameter_lr_dict = []
        for param_group in self.optimizer.param_groups:
            self.parameter_lr_dict.append(param_group["lr"])
        self.optim_param = {"init_lr":0.00003, "gamma":0.3, "power":0.75, "stepsize":3000}
        for val in optim_param:
            self.optim_param[val] = optim_param[val]

        if self.trade_off > 0:
            # initialize covariance matrix
            self.task_cov = torch.eye(num_tasks)
            self.class_cov = torch.eye(output_num)
            self.feature_cov = torch.eye(self.bottleneck_size)

            self.task_cov_var = Variable(self.task_cov).cuda()
            self.class_cov_var = Variable(self.class_cov).cuda()
            self.feature_cov_var = Variable(self.feature_cov).cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.iter_num = 1
       

    def optimize_model(self, input_list, label_list):
        # update learning rate
        current_lr = self.optim_param["init_lr"] * (self.optim_param["gamma"] ** (self.iter_num // self.optim_param["stepsize"]))
        i = 0
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr * self.parameter_lr_dict[i]
            i += 1

        # classification loss
        for i in xrange(self.num_tasks):
            self.networks[i].train(True)
        self.shared_layers.train(True)
        batch_size = input_list[0].size(0)

        self.optimizer.zero_grad()
        concat_input = torch.cat(input_list, dim=0)
        feature_out = self.shared_layers(concat_input)
        output_list = [self.networks[i](feature_out.narrow(0, i*batch_size, batch_size)) for i in xrange(self.num_tasks)]
        losses = [self.criterion(output_list[i], label_list[i]) for i in xrange(self.num_tasks)]
        classifier_loss = sum(losses)

        # multitask loss
        if self.trade_off > 0:                   
            weight_size = self.networks[0].module[-1].weight.size()
            all_weights = [self.networks[i].module[-1].weight.view(1, weight_size[0], weight_size[1]) for i in xrange(self.num_tasks)]
            weights = torch.cat(all_weights, dim=0).contiguous()    
   
            multi_task_loss = tensor_op.MultiTaskLoss(weights, self.task_cov_var, self.class_cov_var, self.feature_cov_var)
            total_loss = classifier_loss + self.trade_off * multi_task_loss
            self.train_cross_loss += classifier_loss.data[0]
            self.train_multi_task_loss += multi_task_loss.data[0]
        else:
            total_loss = classifier_loss
            self.train_cross_loss += classifier_loss.data[0]
        # update network parameters
        total_loss.backward()
        self.optimizer.step()

        if self.trade_off > 0 and self.iter_num % self.cov_update_freq == 0:
            # get updated weights
            weight_size = self.networks[0].module[-1].weight.size()
            all_weights = [self.networks[i].module[-1].weight.view(1, weight_size[0], weight_size[1]) for i in xrange(self.num_tasks)]
            weights = torch.cat(all_weights, dim=0).contiguous() 

            # update cov parameters
            temp_task_cov_var = tensor_op.UpdateCov(weights.data, self.class_cov_var.data, self.feature_cov_var.data)

            
            #temp_class_cov_var = tensor_op.UpdateCov(weights.data.permute(1, 0, 2).contiguous(), self.task_cov_var.data, self.feature_cov_var.data)
            #temp_feature_cov_var = tensor_op.UpdateCov(weights.data.permute(2, 0, 1).contiguous(), self.task_cov_var.data, self.class_cov_var.data)

            
            # task covariance
            u, s, v = torch.svd(temp_task_cov_var)
            s = s.cpu().apply_(self.select_func).cuda()
            self.task_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
            this_trace = torch.trace(self.task_cov_var)
            if this_trace > 3000.0:        
                self.task_cov_var = Variable(self.task_cov_var / this_trace * 3000.0).cuda()
            else:
                self.task_cov_var = Variable(self.task_cov_var).cuda()
            # uncomment to use the other two covariance
            '''
            # class covariance
            u, s, v = torch.svd(temp_class_cov_var)
            s = s.cpu().apply_(self.select_func).cuda()
            self.class_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
            this_trace = torch.trace(self.class_cov_var)
            if this_trace > 3000.0:        
                self.class_cov_var = Variable(self.class_cov_var / this_trace * 3000.0).cuda()
            else:
                self.class_cov_var = Variable(self.class_cov_var).cuda()
            # feature covariance
            u, s, v = torch.svd(temp_feature_cov_var)
            s = s.cpu().apply_(self.select_func).cuda()
            temp_feature_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
            this_trace = torch.trace(temp_feature_cov_var)
            if this_trace > 3000.0:        
                self.feature_cov_var += 0.0003 * Variable(temp_feature_cov_var / this_trace * 3000.0).cuda()
            else:
                self.feature_cov_var += 0.0003 * Variable(temp_feature_cov_var).cuda()
            '''
        self.iter_num += 1
        if self.iter_num % self.print_interval == 0:
            self.train_total_loss = self.train_cross_loss + self.train_multi_task_loss
            print("Iter {:05d}, Average Cross Entropy Loss: {:.4f}; Average MultiTask Loss: {:.4f}; Average Training Loss: {:.4f}".format(self.iter_num, self.train_cross_loss / float(self.print_interval), self.train_multi_task_loss / float(self.print_interval), self.train_total_loss / float(self.print_interval)))
            self.file_out.write("Iter {:05d}, Average Cross Entropy Loss: {:.4f}; Average MultiTask Loss: {:.4f}; Average Training Loss: {:.4f}\n".format(self.iter_num, self.train_cross_loss / float(self.print_interval), self.train_multi_task_loss / float(self.print_interval), self.train_total_loss / float(self.print_interval)))
            self.file_out.flush()
            self.train_cross_loss = 0
            self.train_multi_task_loss = 0
            self.train_total_loss = 0

       
    def test_model(self, input_, i):
        self.shared_layers.train(False)
        self.networks[i].train(False)
        output = self.networks[i](self.shared_layers(input_))
        return output 
