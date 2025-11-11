"""生成 GMNN 所需的网络与标签文件/异构图"""

import json
import re
import numpy as np
import os
from model import *

device = 'cuda:0'
FILE = '../Data/electricity/e10kv2tl.json'
EFILE = '../Data/electricity/all_dict_correct_11_aoi.json'
BSFILE = '../Data/China_BS/bs_relation_combine.json'
ELE2BS_FILE = '../Data/China_BS/ele2bs_combined.json'
#电力网络 (ele) 和基站 (bs) 之间的连接关系（例如，哪个电力节点为哪个基站供电）
TFILE1 = '../Data/road/road_junc_map.json'
TFILE2 = '../Data/road/road_type_map.json'
TFILE3 = '../Data/road/tl_id_road2elec_map.json'
#道路节点 ↔ 电力节点 的跨层依赖（例如，路口&信号控制设施依赖电力节点供电）
AOIFILE = '../Data/China_BS/aoi_combined.json'
ept = '../generate_failure_cascade/embedding/elec_feat.pt'
tpt = '../generate_failure_cascade/embedding/tra_feat.pt'
bspt = '../generate_failure_cascade/embedding/bs_feat.pt'
bpt = ('../generate_failure_cascade/embedding/bifeatures_aoi/bi_elec_feat.pt',
       '../generate_failure_cascade/embedding/bifeatures_aoi/bi_tra_feat.pt',
       '../generate_failure_cascade/embedding/bifeatures_aoi/bi_bs_feat.pt',
       '../generate_failure_cascade/embedding/bifeatures_aoi/bi_aoi_feat.pt')

EMBED_DIM = 64 #每个节点会被表示成长度为 64 的向量
HID_DIM = 128 #图神经网络隐藏层的维度
FEAT_DIM = 64 #输入特征维度（节点原始特征的维度）
KHOP = 5 #图卷积或邻域扩展的最大跳数（感受野半径）,模型会看每个节点周围5跳内的节点信息
BASE = 100000000 #节点编号基数，用于区分不同子图的节点编号，避免冲突,这里暂时没直接用到它

bsgraph = BSGraph(file=BSFILE,
                  embed_dim=EMBED_DIM,
                  hid_dim=HID_DIM,
                  feat_dim=FEAT_DIM,
                  khop=KHOP,
                  epochs=500,
                  pt_path=ept)  #待改：应该是bspt，这里暂用 ept 只是先跑通

egraph = ElecGraph(file=EFILE,
                   embed_dim=EMBED_DIM,
                   hid_dim=HID_DIM,
                   feat_dim=FEAT_DIM,
                   khop=KHOP,
                   epochs=500,
                   pt_path=ept)

tgraph = TraGraph(file1=TFILE1, file2=TFILE2, file3=TFILE3,
                  embed_dim=EMBED_DIM,
                  hid_dim=HID_DIM,
                  feat_dim=FEAT_DIM,
                  khop=KHOP,
                  epochs=300,
                  r_type='tertiary',
                  pt_path=tpt)

aoigraph = AOIGraph(file=AOIFILE,
                    embed_dim=EMBED_DIM,
                    hid_dim=HID_DIM,
                    feat_dim=FEAT_DIM,
                    khop=KHOP,
                    epochs=500,
                    pt_path=ept)

bigraph = Bigraph(efile=EFILE,
                  tfile1=TFILE1, tfile2=TFILE2, tfile3=TFILE3,
                  file=FILE,
                  bsfile=BSFILE,
                  ele2bsfile=ELE2BS_FILE,
                  aoifile=AOIFILE,
                  embed_dim=EMBED_DIM,
                  hid_dim=HID_DIM,
                  feat_dim=FEAT_DIM,
                  subgraph=(bsgraph, egraph, tgraph, aoigraph),
                  khop=KHOP,
                  epochs=1400,
                  r_type='tertiary',
                  pt_path=bpt)
    #合并为异构大图,Bigraph 组合四个子图与跨层关系
print(1)
# 下面建立索引
bi_nxgraph = bigraph.nxgraph
node_dict = {}
for k, v in bigraph.node_list.items():
    node_dict[v] = k
# with open('./GMNN_data/net.txt', 'a') as f:
#     for (u, v) in tqdm(bi_nxgraph.edges):
#         u = node_dict[u]
#         v = node_dict[v]
#         f.write('{0}\t{1}\t{2}\n'.format(u, v, 1))
#（可选）导出 GMNN 的 net/feature 文件
#
# with open('./GMNN_data/feature.txt', 'a') as f:
#     index = 0
#     for node_type in ['power', 'junc', 'bs', 'aoi']:
#         features = bigraph.feat[node_type]
#         for node in trange(len(features)):
#             feat = features[node]
#             index_value_list = [(i, v) for i, v in enumerate(feat.view(-1))]
#             str_feat = ' '.join(['{}:{}'.format(i, v) for i, v in index_value_list])
#             f.write('{0}\t{1}\n'.format(index, str_feat))
#             index += 1


fpath = '../Data/ruin_cascades_for_different_type_and_size/500kv/cases'
# 遍历所有案例文件，生成训练样本与标签
for filename in os.listdir(fpath):
    with open('{0}/{1}'.format(fpath, filename), 'r') as f:
        data = json.load(f)
    source_nodes = data['source']
    source_nodes = [node_dict[v] for v in source_nodes]
    # 训练输入为触发节点列表(故障源头节点)
    train_str = '\n'.join([str(x) for x in source_nodes])
    label_str = ''
    # label 文件要求对所有节点给出标签：1 表示触发节点，-1 表示未知
    #GMNN 语义：只把触发源标成 1，其它节点不设为 0，而是 -1（未知），让模型在图上传播/推断
    for index, node in enumerate(range(len(node_dict))):
        if not node in source_nodes:
            label_str = label_str + '{0}\t{1}\n'.format(index, -1)
        else:
            label_str = label_str + '{0}\t{1}\n'.format(index, 1)
    digits = re.findall('\d', filename)
    node_id = ''.join(digits) # 从文件名中提取数字作为节点 ID
    with open('./GMNN_data/train/train_{0}.txt'.format(node_id), 'w') as f:
        f.write(train_str)
    with open('./GMNN_data/label/label_{0}.txt'.format(node_id), 'w') as f:
        f.write(label_str)
