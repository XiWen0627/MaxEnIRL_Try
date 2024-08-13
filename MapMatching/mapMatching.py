# _*_ coding : utf-8 _*_
# @Time : 2023-11-03 16:45
# @Author : PeterPan
# @File : 00_mapMatching
# @Project : 路网匹配
###### 需要获取原轨迹点在图数据中的节点编号 lastidx 的编号规则 #######
###### 此步需要在构建道路网络步进行 #############
import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap
import warnings

warnings.filterwarnings("ignore")

def construct_road_network(mapcon, shapefile, coordinate='epsg:4547'):
    '''
    本函数的目的是为了构建有待于路网匹配的道路网络
    *** 由于本函数采用相对路径读取文件, 因此使用前需要设置工作空间
    *** 输入的道路路网应该不包含多部件要素

    :param mapcon: 初始化后的 InMemMap 对象
    :param shapefile: 待匹配的路网数据
    :param coordinate: 待匹配路网数据的投影坐标系
    :return: 可以用于路网匹配的图数据
    '''

    print('开始构建路网网络')
    node_file_path = r'./node.txt'
    edge_file_path = r'./edge.txt'
    nodefile = open(node_file_path, mode='w', encoding='utf-8')
    edgefile = open(edge_file_path, mode='w', encoding='utf-8')

    gdf = gpd.read_file(shapefile)  # shp转gdf
    gdf.crs = {'init': coordinate}

    # 添加节点
    nodes = {}
    node_idx = 0

    # 防止因输入多部件要素而报错
    try:
        for lineid, row in gdf.iterrows():
            nodes_seq = row['geometry'].coords[:]  # 获取LineString对象的所有折点的坐标，返回元组列表
            # print(row['OBJECTID'])
            # 将node 的idx 与路段的objectid对应
            for node in nodes_seq:
                if nodes.get(node, -1) == -1:
                    nodes[node] = node_idx
                    node_idx += 1
    except:
        print('-'*30)
        print('报错信息: 输入要素不应该多部件要素。\n请检查输入要素的构成')
        print('-'*30)

    for k, v in nodes.items():
        mapcon.add_node(v, (k[1], k[0]))
        nodefile.write(f'{v},{k[1]},{k[0]}\n')

    # 添加边
    edges = set()
    for lineid, row in gdf.iterrows():
        nodes_seq = row['geometry'].coords[:]
        for i in range(len(nodes_seq) - 1):
            edges.add((nodes.get(nodes_seq[i]), nodes.get(nodes_seq[i + 1])))
            edges.add((nodes.get(nodes_seq[i + 1]), nodes.get(nodes_seq[i])))  # 骑行不受行驶方向限制，因此构建无向边

    for edge in edges:
        mapcon.add_edge(edge[0], edge[1])
        edgefile.write(f'{edge}\n')

    nodefile.close()
    edgefile.close()
    print('路网网络构建完成')
    return mapcon

def load_road_network(network_dir, map_con):
    """
    本函数的目的是为了从文件中加载节点与边文件

    :param network_dir:
    :param map_con:
    :return: 可以用于匹配的图数据
    """
    with open(fr'{network_dir}\node.txt', mode='r', encoding='utf-8') as f_node:
        for line in f_node:
            node = line.strip('\n').split(',')
            map_con.add_node(int(node[0]), (float(node[1]), float(node[2])))
    with open(fr'{network_dir}\edge.txt', mode='r', encoding='utf-8') as f_edge:
        for line in f_edge:
            edge = line.strip('\n').split(',')
            map_con.add_edge(int(edge[0].strip('(')), int(edge[1].strip().strip(')')))
    print('路网网络载入完成')
    return map_con

def to_pixels(lat, lon=None):
    if lon is None:
        lat, lon = lat[0], lat[1]
    return lon, lat


if __name__ == '__main__':
    os.chdir(r'G:\BaiduNetdiskDownload\共享单车轨迹\共享单车轨迹')
    my_shapefile = r'./地理数据/roadLG.shp'

    # ----------------------------------------------------------------------------------------- #
    # ---------------------------------- 构建路网匹配器 ------------------------------------------ #
    # ----------------------------------------------------------------------------------------- #
    # 初始化 InMemMap 对象
    map_init = InMemMap("longgangmap", use_latlon=False, use_rtree=False, index_edges=True, crs_xy=4547)
    map_con = construct_road_network(shapefile=my_shapefile, mapcon=map_init)
    # print(map_con)    # InMemMap(longgangmap, size=3715)
    # 构造匹配器
    matcher = DistanceMatcher(map_con,
                              max_dist=50,
                              max_dist_init=50,
                              min_prob_norm=0.01,
                              non_emitting_length_factor=0.75,
                              obs_noise=50,
                              obs_noise_ne=50,
                              dist_noise=50,
                              max_lattice_width=50,
                              non_emitting_states=True,
                              avoid_goingback=True
                              )

    # ----------------------------------------------------------------------------------------- #
    # --------------------------------------  进行路网匹配 --------------------------------------- #
    # ----------------------------------------------------------------------------------------- #
    # 创建新文件以储存匹配结果
    matchedFilePath = r'./数据处理/matched1.txt'
    # 新建文件以储存待匹配的轨迹点
    matchFile = open(matchedFilePath, mode='w', encoding='utf-8')

    # 轨迹点序列的生成
    track_df_ini = pd.read_csv(r'./数据处理/00_rawData.txt', sep=',')
    print('轨迹点的总数')
    print(track_df_ini.shape)
    print('_'*30)
    breakPoints = np.unique(track_df_ini['breakPt'])    # 经过切分之后的轨迹点
    # breakPoints = np.unique(track_df_ini['orderid'])    # 经过切分之后的轨迹点

    # 对于一条轨迹中的所有节点对
    for breakPoint in breakPoints:
        track_df = track_df_ini[track_df_ini['breakPt'] == breakPoint]    # 选取目标路径
        track_df.sort_values(by=['breakPt', 'ttime'], ascending=[True, True], inplace=True)  # dataframe按多字段排序

        # 排序之后获得值字段
        IDList = np.unique(track_df['FID'].values)    # 每个点的唯一标识字段
        # print(IDList)

        print('本条轨迹共有{}个轨迹点有待匹配'.format(len(IDList)))
        print('_'*30)

        track = []    # 结果为 108
        # 将轨迹点写入到列表之中
        for breakPt, x, y in zip(track_df['breakPt'], track_df['x'], track_df['y']):
            track.append((y, x))

        # 已经获得轨迹点序列
        states, _ = matcher.match(track)
        lat_nodes = matcher.lattice_best    # 经过匹配后的结果
        # 开始时间
        startTime = time.time()

        i = 0    # 作为索引记录轨迹与节点编号之间的唯一对应关系以应对一对多的情况
        for idx, m in enumerate(lat_nodes):
            # print('_'*60)
            lat, lon = m.edge_m.pi[:2]
            lat2, lon2 = m.edge_o.pi[:2]
            x, y = to_pixels(lat, lon)    # 匹配好轨迹点的横纵坐标
            x2, y2 = to_pixels(lat2, lon2)    # 原坐标

            # 需要获得当前节点的 ID
            # 查看该种数据类型能否输出 o 点的编号 查看可视化代码应该是可以输出的
            # 该文件如何与源文件中的节点编号进行对应？
            if m.edge_o.is_point():    # 是否找到了最佳匹配记录
                # 将 ID 字段与轨迹字段写入到新的文件之中
                # 可用于查询该点的信息
                # {是否是一对多的情况}{节点FID编号}{匹配后的x坐标}{匹配后的y坐标}
                matchFile.write(f'{"True"},{IDList[i]},{breakPoint},{x},{y}\n')
                i += 1
                # print(i)
                # print('本条轨迹的ID为{}'.format(IDList[i]))
            else:
                matchFile.write(f'{"False"},{IDList[i]},{breakPoint},{x},{y}\n')
                # print(i)
        # 结束时间
        endTime = time.time()
        runTime = endTime - startTime
        print('_' * 15)
        print('编号为{}条轨迹的路网匹配已经完成'.format(breakPoint))
        print("程序的运行时间为：%f秒" % runTime)


    print('所有节点都已经匹配完成并且写入到文件之中')
    matchFile.close()