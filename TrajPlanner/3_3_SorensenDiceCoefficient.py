import os
import pandas as pd
import numpy as np
import json
from multiset import Multiset

# 设置 numpy 科学计数法
np.set_printoptions(suppress=True, threshold=5000)
# 设置 pandas 的格式
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.float_format', lambda x: '%.8f' % x)

SP_file = r'./shortestPath.json'
with open(SP_file, 'r') as F:
    SP_Data = json.load(F)

trajFile = r'F:\BaiduNetdiskDownload\共享单车轨迹\共享单车轨迹\01_研究生毕业论文\1_2_DataPreProcess/BanTianTrajCom.txt'
trajData = pd.read_csv(trajFile, encoding='utf-8')

CPC_File = r'./CPC.txt'
with open(CPC_File, 'w') as F1:
    F1.write(f'{"TrajID"},{"CPC"}\n')

    for key in SP_Data.keys():
        SP_set = Multiset(SP_Data[key][0])
        trajData_info = trajData[trajData['TrajID'] == int(key)].sort_values(by='timeStamp',
                                                                             ascending=True)

        trajList = list(trajData_info['LG_ID'].astype(int).values)
        trajSet = Multiset(trajList)
        sdc = (2 * len(SP_set & trajSet)) / (len(SP_set)+len(trajSet))
        F1.write(f'{key},{sdc}\n')
        print('编号为{}的轨迹与最短路径的相似性已经计算完成, 其 SDC 为{}'.format(key, sdc))

    F1.close()
