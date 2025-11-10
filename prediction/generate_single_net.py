"""将综合网络拆分为单独子网文件，新增中文注释说明流程。"""

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
    # 遍历原始网络文件，根据节点编号范围划分子网络
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
    # 将电力子图写入文件
    f.write(elec)
with open('./GMNN_data/junc_net.txt', 'w') as f:
    # 将交通节点子图写入文件
    f.write(junc)
with open('./GMNN_data/bs_net.txt', 'w') as f:
    # 将基站子图写入文件
    f.write(bs)
