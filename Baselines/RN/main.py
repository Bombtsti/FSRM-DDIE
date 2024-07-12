#coding:utf-8
import pickle
import warnings

from graph2vector import SSI_DDI, get_data

warnings.filterwarnings("ignore")
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pandas as pd
import sqlite3
#import torchsnooper
#import task_generator as tg
import task_generator_META_DDIE as tg
import os
import math
import argparse
import random
import time
# from smiles2vector import smiles2vector
from dde_config import dde_NN_config
from dde_torch import dde_NN_Large_Predictor
#10个样本的事件为175
#5个样本截止为197
CLASS_NUM=175
NUM_WAYS=5
Support_NUM_PER_CLASS=1
QUERY_NUM_PER_CLASS=4
dropoutRate=0.5
FEATURE_DIMENSION=64 #64
FLAT=2048
RELATION_DIMENSION=8
LEARNING_RATE=0.0001
EPISODE=100000
TEST_EPISODE=5000
SMILE_SHAPE=1706
GPU=0
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self):
        super(RelationNetwork, self).__init__()
        self.reluDrop = nn.Sequential(nn.ReLU(),nn.Dropout(dropoutRate))
        self.layer1 = nn.Sequential(
                        nn.Conv1d(1,32,kernel_size=5,padding=1),
                        nn.BatchNorm1d(32, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool1d(5))
        self.layer2 = nn.Sequential(
                        nn.Conv1d(32,32,kernel_size=5,padding=1),
                        nn.BatchNorm1d(32, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool1d(5))
        #self.fc1 = nn.Sequential(nn.Linear(832,256),self.reluDrop,nn.BatchNorm1d(256)) # for 343 input dimension
        #self.fc1 = nn.Sequential(nn.Linear(5504,256),self.reluDrop,nn.BatchNorm1d(256)) # for 2161 input dimension
        self.fc1 = nn.Sequential(nn.Linear(224,64), self.reluDrop, nn.BatchNorm1d(64))  # for 1722 input dimension
        # self.fc2 = nn.Sequential(nn.Linear(128,64),self.reluDrop,nn.BatchNorm1d(64))
        # self.fc3 = nn.Sequential(nn.Linear(1024,256),self.reluDrop,nn.BatchNorm1d(256))
        self.fc4 = nn.Linear(64,1)

    def forward(self,x):
        x = torch.reshape(x,(-1,1,x.shape[1]))
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        # out = self.fc2(out)
        #out = self.fc3(out)
        out = F.sigmoid(self.fc4(out))
        return out

# class RelationNetwork(nn.Module):
#     def __init__(self):
#         super(RelationNetwork, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv1d(1,32,kernel_size=5,padding=1),
#             nn.BatchNorm1d(32, momentum=1, affine=True),
#             nn.ReLU(),
#             nn.MaxPool1d(5)
#         )
#         self.layer2 = nn.Sequential(
#             nn.Conv1d(32,32,kernel_size=5,padding=1),
#             nn.BatchNorm1d(32, momentum=1, affine=True),
#             nn.ReLU(),
#             nn.MaxPool1d(5)
#         )
#         self.encoder = nn.Sequential(
#             self.layer1,
#             self.layer2
#         )
#     def euclidean_dist(self, x, y):
#         n = x.size(0)
#         m = y.size(0)
#         d = x.size(1)
#         assert d == y.size(1)
#         # x = x.unsqueeze(1).expand(n, m, d)
#         # y = y.unsqueeze(0).expand(n, m, d)
#         # distances = torch.pow(x-y,2).sum(2)
#         distances = torch.sqrt(torch.sum((x[:, None, :] - y[None, :, :]) ** 2, dim=2))
#         return distances
#
#     def forward(self,x):
#         x = torch.reshape(x, (-1, 1, x.shape[1]))
#         z = self.encoder(x)
#         z = z.view(z.size(0), -1)
#         z_proto = z[:5].view(5, 1, -1).mean(1)
#         zq = z[5:]
#         dists = self.euclidean_dist(zq, z_proto)
#         p_y = F.log_softmax(-dists, dim=1)
#         return p_y

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0]  * m.out_channels # m.kernel_size[1]针对Conv2D
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

import sys
class Logger(object):
  def __init__(self, filename="Default.log"):
    self.terminal = sys.stdout
    self.log = open(filename, "a")
  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)
  def flush(self):
    pass

def main():
    excludeLabel=1
    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger(str(excludeLabel)+'-db86-rn-rare-5way.txt')

    #read data
    config = dde_NN_config()

    with open('./drugbank_drug_graph.pkl', 'rb') as f:
        drug_graph = pickle.load(f)

    #init neural network
    print("init neural networks")
    model_nn = SSI_DDI(55, 64)
    relation_network = RelationNetwork()

    model_nn=model_nn.to(device)
    relation_network.apply(weights_init)
    relation_network=relation_network.to(device)


    model_nn_optim = torch.optim.Adam(model_nn.parameters(), lr = LEARNING_RATE)
    model_nn_scheduler = StepLR(model_nn_optim,step_size=100000,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=100000, gamma=0.5)

    # if os.path.exists(str("models1005/model_nn_" + str(NUM_WAYS) +"ways_" + str(Support_NUM_PER_CLASS) +"shot"+str(excludeLabel)+"mag"+str(config['magnify_factor'])+str(config['input_dim'])+".pkl")):
    #     model_nn.load_state_dict(torch.load(str("models1005/model_nn_" + str(NUM_WAYS) +"ways_" + str(Support_NUM_PER_CLASS) +"shot"+str(excludeLabel)+"mag"+str(config['magnify_factor'])+str(config['input_dim'])+".pkl"),map_location=device))
    #     print("load CASTER success")
    # if os.path.exists(str("models1005/relation_network_"+ str(NUM_WAYS) +"ways_" + str(Support_NUM_PER_CLASS) +"shot"+str(excludeLabel)+"mag"+str(config['magnify_factor'])+str(config['input_dim'])+".pkl")):
    #     relation_network.load_state_dict(torch.load(str("models1005/relation_network_"+ str(NUM_WAYS) +"ways_" + str(Support_NUM_PER_CLASS) +"shot"+str(excludeLabel)+"mag"+str(config['magnify_factor'])+str(config['input_dim'])+".pkl"),map_location=device))
    #     print("load relation network success")

    #start training
    last_accuracy=0.0
    seen_acc = 0.0
    total_train_rewards = 0
    start = time.time()

    for episode in range(EPISODE):
        model_nn_scheduler.step(episode)
        relation_network_scheduler.step(episode)
        task = tg.MetaDDIETask(CLASS_NUM, NUM_WAYS, Support_NUM_PER_CLASS, QUERY_NUM_PER_CLASS,"train",excludeLabel)
        support_dataloader = tg.get_data_loader(task, num_per_class=Support_NUM_PER_CLASS, split="train", shuffle=False)
        query_dataloader = tg.get_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)
        support_drug1, support_drug2, support_labels = support_dataloader.__iter__().__next__()
        query_drug1, query_drug2, query_labels = query_dataloader.__iter__().__next__()


        # ******** graph2vector ***********
        support_h_list = []
        support_t_list = []
        query_h_list = []
        query_t_list = []
        for i in range(Support_NUM_PER_CLASS*NUM_WAYS):
            smol1_graph_data = drug_graph[support_drug1[i]]
            smol2_graph_data = drug_graph[support_drug2[i]]
            support_h_list.append(smol1_graph_data)
            support_t_list.append(smol2_graph_data)
            # support_sample_drugs = np.vstack((support_sample_drugs,np.reshape(smiles2vector(smile[support_drug1[i]],smile[support_drug2[i]]),(1,-1))))
        for i in range(QUERY_NUM_PER_CLASS*NUM_WAYS):
            qmol1_graph_data = drug_graph[query_drug1[i]]
            qmol2_graph_data = drug_graph[query_drug2[i]]
            query_h_list.append(qmol1_graph_data)
            query_t_list.append(qmol2_graph_data)
            # query_sample_drugs = np.vstack((query_sample_drugs,np.reshape(smiles2vector(smile[query_drug1[i]],smile[query_drug2[i]]),(1,-1))))

        support_loader = get_data(support_h_list,support_t_list)
        query_loader = get_data(query_h_list,query_t_list)
        mag_support_feature,repr_support_before,repr_support_after = model_nn(support_loader)
        mag_query_feature,repr_query_before,repr_query_after = model_nn(query_loader)
        # ******** graph2vector ***********


        support_features_ext = mag_support_feature.unsqueeze(0).repeat(QUERY_NUM_PER_CLASS * NUM_WAYS, 1, 1)
        query_features_ext = mag_query_feature.unsqueeze(0).repeat(Support_NUM_PER_CLASS * NUM_WAYS, 1, 1)
        query_features_ext = torch.transpose(query_features_ext, 0, 1)


        relation_pairs = torch.cat((support_features_ext,query_features_ext),2).view(-1,config['input_dim']*2)
        # relation_pairs = torch.cat((mag_support_feature,mag_query_feature),dim=0)
        relation_pairs = relation_pairs.to(device)
        relations = relation_network(relation_pairs).view(-1,NUM_WAYS)

        mse = nn.MSELoss().to(device)

        query_labels_array=np.array(query_labels.view(QUERY_NUM_PER_CLASS*NUM_WAYS))
        query_labels_array = (np.arange(query_labels_array.max() + 1) == query_labels_array[:, None]).astype(dtype='float32')
        one_hot_labels = Variable(torch.from_numpy(query_labels_array).to(device))
        query_labels=query_labels.to(device)

        _, predict_labels = torch.max(relations.data, 1)
        train_rewards = [1 if predict_labels[j] == query_labels[j] else 0 for j in range(NUM_WAYS*QUERY_NUM_PER_CLASS)]
        total_train_rewards += np.sum(train_rewards)

        loss_c = mse(relations, one_hot_labels).to(device)
        loss_s = mse(repr_support_before,repr_support_after).to(device)
        loss_q = mse(repr_query_before,repr_query_after).to(device)
        loss = loss_c + 0.1 * loss_s + 0.1 * loss_q


        #feature_encoder.zero_grad()
        model_nn.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm(model_nn.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(), 0.5)


        model_nn_optim.step()
        relation_network_optim.step()
        if (episode + 1) % 500 == 0:
            end = time.time()
            print("episode:", episode + 1, "loss", loss.data)
            train_accuracy = total_train_rewards / 500.0 / NUM_WAYS / QUERY_NUM_PER_CLASS
            print("train_acc:",train_accuracy)
            print("use:" + str(end - start))
            start = time.time()
            if train_accuracy > seen_acc:
                torch.save(model_nn.state_dict(), str(
                    "models1005/model_nn_" + str(NUM_WAYS) + "ways_" + str(Support_NUM_PER_CLASS) + "shot" + str(
                        excludeLabel) + "mag" + str(config['magnify_factor']) + str(config['input_dim']) + "seen.pkl"))
                torch.save(relation_network.state_dict(), str(
                    "models1005/relation_network_" + str(NUM_WAYS) + "ways_" + str(
                        Support_NUM_PER_CLASS) + "shot" + str(excludeLabel) + "mag" + str(
                        config['magnify_factor']) + str(config['input_dim']) + "seen.pkl"))
                seen_acc = train_accuracy

            total_train_rewards=0

        if (episode + 1) % 5000 == 0: #1000
            # test
            print("Testing...")
            total_rewards = 0
            for i in range(TEST_EPISODE): #TEST_EPISODE
                task = tg.MetaDDIETask(CLASS_NUM, NUM_WAYS, Support_NUM_PER_CLASS, Support_NUM_PER_CLASS,"test",excludeLabel)
                support_dataloader = tg.get_data_loader(task, num_per_class=Support_NUM_PER_CLASS, split="train",
                                                        shuffle=False)
                query_dataloader = tg.get_data_loader(task, num_per_class=Support_NUM_PER_CLASS, split="test",
                                                      shuffle=True)
                support_drug1, support_drug2, support_labels = support_dataloader.__iter__().__next__()
                query_drug1, query_drug2, query_labels = query_dataloader.__iter__().__next__()

                # ******** graph2vector ***********
                support_h_list = []
                support_t_list = []
                query_h_list = []
                query_t_list = []
                for i in range(Support_NUM_PER_CLASS * NUM_WAYS):
                    smol1_graph_data = drug_graph[support_drug1[i]]
                    smol2_graph_data = drug_graph[support_drug2[i]]
                    support_h_list.append(smol1_graph_data)
                    support_t_list.append(smol2_graph_data)
                    # support_sample_drugs = np.vstack((support_sample_drugs,np.reshape(smiles2vector(smile[support_drug1[i]],smile[support_drug2[i]]),(1,-1))))
                for i in range(Support_NUM_PER_CLASS * NUM_WAYS):
                    qmol1_graph_data = drug_graph[query_drug1[i]]
                    qmol2_graph_data = drug_graph[query_drug2[i]]
                    query_h_list.append(qmol1_graph_data)
                    query_t_list.append(qmol2_graph_data)
                    # query_sample_drugs = np.vstack((query_sample_drugs,np.reshape(smiles2vector(smile[query_drug1[i]],smile[query_drug2[i]]),(1,-1))))

                support_loader = get_data(support_h_list, support_t_list)
                query_loader = get_data(query_h_list, query_t_list)
                mag_support_feature, repr_support_before, repr_support_after = model_nn(support_loader)
                mag_query_feature, repr_query_before, repr_query_after = model_nn(query_loader)
                # ******** graph2vector ***********


                support_features_ext = mag_support_feature.unsqueeze(0).repeat(Support_NUM_PER_CLASS * NUM_WAYS, 1, 1)
                query_features_ext = mag_query_feature.unsqueeze(0).repeat(Support_NUM_PER_CLASS * NUM_WAYS, 1, 1)
                query_features_ext = torch.transpose(query_features_ext, 0, 1)

                support_labels = support_labels.to(device)
                query_labels = query_labels.to(device)


                relation_pairs = torch.cat((support_features_ext, query_features_ext), 2).view(-1, config['input_dim']*2)
                # relation_pairs =torch.cat((mag_support_feature,mag_query_feature),dim=0)

                relation_pairs = relation_pairs.to(device)
                relations = relation_network(relation_pairs).view(-1, NUM_WAYS)
                _, predict_labels = torch.max(relations.data, 1)
                rewards = [1 if predict_labels[j] == query_labels[j] else 0 for j in range(NUM_WAYS)]

                total_rewards += np.sum(rewards)
                #print(sum(rewards))

            test_accuracy = total_rewards / 1.0 / NUM_WAYS / TEST_EPISODE

            print("test accuracy:", test_accuracy)
            print(relations)

            if test_accuracy>last_accuracy:

                torch.save(model_nn.state_dict(),str("models1005/model_nn_"+str(NUM_WAYS)+"ways_"+str(Support_NUM_PER_CLASS)+"shot"+str(excludeLabel)+"mag"+str(config['magnify_factor'])+str(config['input_dim'])+".pkl"))
                torch.save(relation_network.state_dict(),str("models1005/relation_network_" + str(NUM_WAYS) + "ways_" + str(Support_NUM_PER_CLASS) + "shot"+str(excludeLabel)+"mag"+str(config['magnify_factor'])+str(config['input_dim'])+".pkl"))

                print("save networks for episode:",episode)

                last_accuracy=test_accuracy

if __name__ == '__main__':
    main()



