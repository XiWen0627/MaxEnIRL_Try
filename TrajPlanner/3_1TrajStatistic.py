# _*_ coding : utf-8 _*_
# @Time : 2024-04-26 16:23
# @Author : PeterPan
# @File : 2_2_1ShortestPath
# @Project : 01_研究生毕业论文
# 对轨迹数据进行探索性数据分析
import os
import numpy as np
import pandas as pd

# dataframe显示设置
pd.set_option('display.max_columns', None)  # 显示所有列
# 设置 pandas 的格式
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)

os.chdir(r'F:\BaiduNetdiskDownload\共享单车轨迹\共享单车轨迹\01_研究生毕业论文\1_2_DataPreProcess')
trajComFile = r'./BanTianTrajCom.txt'
trajComData = pd.read_csv(trajComFile, encoding='utf-8')
trajComData['LGID'] = trajComData['LG_ID'].apply(lambda x: int(x))
print(trajComData)
print('_'*30)

roadInfoFile = r'BanTian_roadLength.csv'
roadInfo = pd.read_csv(roadInfoFile, encoding='utf-8')
print(roadInfo)
print('_'*30)

trajComDataMerge = pd.merge(trajComData, roadInfo, how='left', on='LGID')
print(trajComDataMerge)

# # print(trajComData)
#
trajInfoFile = r'./Bantian_TrajInfo_1.txt'
F = open(trajInfoFile, mode='w', encoding='utf-8')
F.write(f'{"TrajID"},{"numDecision"},{"totalLength"},{"timeStamp"}\n')

for TrajID in np.unique(trajComDataMerge['TrajID'].values):
    print('当前的轨迹 ID 为{}'.format(TrajID))

    # TrajData
    TrajData = trajComDataMerge[trajComDataMerge['TrajID'] == TrajID].sort_values(by='ttime', ascending=True)
    print('当前的轨迹共经过了{}条路段'.format(TrajData.shape[0]))

    timeStamp = int(TrajData.iloc[0]['timeStamp'])
    print('当前轨迹所在的时间为{}'.format(timeStamp))

    total_Length = TrajData['roadLength'].values.sum()
    print('当前轨迹共经过{}米'.format(total_Length))

    F.write(f'{int(TrajID)},'
            f'{TrajData.shape[0]},'
            f'{total_Length},'
            f'{timeStamp}\n')

F.close()