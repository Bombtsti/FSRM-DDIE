import pickle
import random
import sqlite3

import torch
from torch_geometric.data import Batch

from graph2vector import SSI_DDI

# 加载数据
conn = sqlite3.connect("../METADDIEdata/event_META_DDIE_5foldLabel.db")
cur=conn.cursor()

com_events = {}
for i in range(1,176):
    temp = cur.execute('select * from event' + str(i))
    temp=temp.fetchall()
    ddis = random.sample(temp,300) if len(temp)>300 else temp
    com_events[i] = ddis

few_events = {}
for i in range(176,205):
    temp = cur.execute('select * from event' + str(i))
    temp=temp.fetchall()
    ddis = temp
    few_events[i] = ddis

rare_events = {}
for i in range(205,228):
    temp = cur.execute('select * from event' + str(i))
    temp=temp.fetchall()
    ddis = temp
    rare_events[i] = ddis

with open('./drug_graph.pkl', 'rb') as f:
    drug_graph = pickle.load(f)

# 加载模型
excludeLabel = 4
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model_nn = SSI_DDI(55,64)
model_nn=model_nn.to(device)
# model_nn.load_state_dict(torch.load(str("models1005/model_nn_5ways_4shot" + str(excludeLabel) + "mag100100.pkl"),map_location=device))

# dataloader
def get_data(h_list, t_list):
    h_loader = Batch.from_data_list(h_list).to(device)
    t_loader = Batch.from_data_list(t_list).to(device)
    data = (h_loader,t_loader)
    return data

com_event_feature = []
for i in range(1,176):
    event = com_events[i]
    h_list = []
    t_list = []
    for i in range(len(event)):
        smol1_graph_data = drug_graph[event[i][1]]
        smol2_graph_data = drug_graph[event[i][2]]
        h_list.append(smol1_graph_data)
        t_list.append(smol2_graph_data)
    data = get_data(h_list,t_list)
    feature,repr_before,repr_after = model_nn(data)
    event_feature = torch.mean(feature,dim=0,keepdim=True)
    com_event_feature.append(event_feature)
com_event_feature = torch.cat(com_event_feature,dim=0)
torch.save(com_event_feature, 'com_event_feature.pt')

