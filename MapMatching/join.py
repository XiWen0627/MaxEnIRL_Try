# _*_ coding : utf-8 _*_
# @Time : 2023-11-15 19:53
# @Author : PeterPan
# @File : 01_Join
# @Project : 路网匹配
# 本步骤应该位于完成路网匹配之后
# 将匹配后的轨迹点按照唯一 ID 连接到匹配前的轨迹点中
# 本脚本的目的是为了将原轨迹点的时间信息匹配至新的轨迹点之上
import pandas as pd
import os
import time

# 设置 pandas 的格式
pd.options.display.precision = 6
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.float_format', lambda x: '%.6f' % x)

os.chdir(r'G:\BaiduNetdiskDownload\共享单车轨迹\共享单车轨迹\数据处理')

# 读取尚未匹配的数据
startTime = time.time()
rawData = pd.read_csv(r'./00_rawData.txt', sep=',')
# Index(['Unnamed: 0', 'FID', 'ID', 'orderid', 'sectionID', 'LID', 'x', 'y',
#        'ttime', 'dis', 'span', 'speed', 'angle', 'dis1', 'breakPt'],
#       dtype='object')
print(rawData.head(5))
endTime1 = time.time()
runTime1 = endTime1 - startTime
print('_'*30)
print('已经完成 raw Data 的读取，运行时间为{}秒'.format(runTime1))

# 读取已经匹配完成的数据
matchData = pd.read_csv(r'./matched2.txt', sep=',', names=['index', 'Bool', 'FID1', 'TrajID', 'PointX', 'PointY'])
# matchData = pd.read_csv(r'./matched.txt', sep=',', names=['Bool', 'FID1', 'TrajID', 'PointX', 'PointY', 'PointX1', 'PointY2'])
# matchData.drop(['index'], axis=0, inplace=True)
print(matchData.columns)
print(matchData.head(5))
endTime2 = time.time()
runTime2 = endTime2-endTime1
print('_'*30)
print('已经完成 matched Data 的读取，运行时间为{}秒'.format(runTime2))

# 按照 FID 将两数据连接在一起
matchDataFinal = matchData.merge(rawData, left_on='FID1', right_on='FID')
print(matchDataFinal.head(5))
endTime3 = time.time()
runTime3 = endTime3-endTime2
print('_'*30)
print('已经完成数据的连接，运行时间为{}秒'.format(runTime3))

# 将每一行的数据写入到新文件之中
matchedFilePath = r'./matched3.txt'
matchFile = open(matchedFilePath, mode='w', encoding='utf-8')
matchDataFinal.apply(lambda x: matchFile.write(f'{x["FID1"]},{x["orderid"]},{x["TrajID"]},{x["PointX"]},{x["PointY"]},{x["ttime"]}\n'), axis=1)    # 将所需字段写入到文件之中
endTime4 = time.time()
runTime4 = endTime4-endTime3
print('已经完成数据的写入，运行时间为{}秒'.format(runTime4))