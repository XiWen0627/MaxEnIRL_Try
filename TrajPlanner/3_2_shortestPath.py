import numpy as np
import pandas as pd
import networkx as nx
import json
import os

# 设置 numpy 科学计数法
np.set_printoptions(suppress=True, threshold=5000)
# 设置 pandas 的格式
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.float_format', lambda x: '%.8f' % x)

os.chdir(r'F:\BaiduNetdiskDownload\共享单车轨迹\共享单车轨迹\01_研究生毕业论文\1_2_DataPreProcess')
dualNet_File = r'./dualNet.txt'

dualNet_Data = pd.read_csv(dualNet_File, encoding='utf-8')    # 读取对偶网络文件
dualNet_Data1 = dualNet_Data.copy()
dualNet_Data2 = pd.merge(dualNet_Data, dualNet_Data1, left_on=['LGID','LGID_1'],right_on=['LGID_1', 'LGID'])

dualNet_Data3 = dualNet_Data2.drop(['LGID_y', 'LGID_1_y'], axis=1)

dualGraph = nx.DiGraph()

dualNet_Data3.apply(lambda x: dualGraph.add_edge(u_of_edge=int(x['LGID_x']),
                                                 v_of_edge=int(x['LGID_1_x']),
                                                 weight=x['Shape_Leng_x']), axis=1)

# 计算最短路径
TrajFile = r'BanTianTrajCom.txt'
TrajData = pd.read_csv(TrajFile, encoding='utf-8')
shortestPath_Dict = {}

for TrajID in np.unique(TrajData['TrajID'].values):
    if TrajID == 0:
        continue
    else:
        ID_Data = TrajData[TrajData['TrajID'] == TrajID].sort_values(by='timeStamp', ascending=True)

        # 计算给定 OD 之间的最短路径
        # 提取 OD
        Origin = int(ID_Data.iloc[0]['LG_ID'])
        destination = int(ID_Data.iloc[-1]['LG_ID'])
        print('起点ID为{}, 终点 ID为 {}'.format(Origin, destination))

        # 最短路径 与 最短路径长度
        pathList = nx.dijkstra_path(dualGraph,
                                    Origin,
                                    destination)

        pathLength = nx.dijkstra_path_length(dualGraph,
                                             Origin,
                                             destination)
        trajList = [pathList, pathLength]
        shortestPath_Dict[int(TrajID)] = trajList

        print('ID为{}的轨迹已经处理完成'.format(TrajID))
        print('_' * 30)

print('正在写入文件之中···')
shortestPath_File = r'./shortestPath.json'

with open(shortestPath_File, 'w') as F:
    json.dump(shortestPath_Dict, F)