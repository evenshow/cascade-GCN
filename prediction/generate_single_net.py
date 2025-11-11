"""将综合网络拆分为单独子网文件，去掉不同子系统之间的耦合边"""
#用于单网络分析
import json
from tqdm import *

elec_data = range(0, 10887)
junc_data = range(10887, 10887 + 4825)
bs_data = range(10887 + 4825, 10887 + 4825 + 20229)
aoi_data = range(10887 + 4825 + 20229, 10887 + 4825 + 20229 + 10533)
elec = ''
junc = ''
bs = ''
aoi = ''
with open('./GMNN_data/net.txt', 'r') as f:
    # 读取综合网络边列表文件 net.txt
    # 这里每一行的形式是源节点s,目标节点t,权重weight
    for line in tqdm(f.readlines()):
        [s, t, _] = line.split('\t')
        s = int(s)
        t = int(t)
        if s in elec_data and t in elec_data:
            elec += '{0}\t{1}\t1\n'.format(s, t)
        elif s in junc_data and t in junc_data:
            junc += '{0}\t{1}\t1\n'.format(s, t)
        elif s in bs_data and t in bs_data:
            bs += '{0}\t{1}\t1\n'.format(s, t)
        elif s in aoi_data and t in aoi_data:
            aoi += '{0}\t{1}\t1\n'.format(s, t)
with open('./GMNN_data/elec_net.txt', 'w') as f:
    # 只将电力子图写入文件
    #举例：(电力节点, 电力节点) → 归入电力子图 ✅
    # (电力节点, 基站节点) → 跨网络边，不属于任何一个子图 → 被忽略
    f.write(elec)
with open('./GMNN_data/junc_net.txt', 'w') as f:
    # 将交通节点子图写入文件
    f.write(junc)
with open('./GMNN_data/bs_net.txt', 'w') as f:
    # 将基站子图写入文件
    f.write(bs)
