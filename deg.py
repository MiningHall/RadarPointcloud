import plotly
import plotly.graph_objs as go
import numpy as np
from numpy import sin,cos,mean,angle,pi,sinc,std,savetxt,rad2deg,arccos,sqrt, var
import matplotlib.pyplot as plt
import sys
import struct
import open3d as o3d
import random

'''
外置参数：
两次滤波的设置
聚类参数设置
聚类滤波设置

'''
def xyz2pcd(points):
    points[:,2] = points[:,2]*100000
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def pcd2xyz(pcd):
    points = np.asarray(pcd.points)
    points[:,2] = points[:,2]*0.00001
    return points

def load_dat(fn):
    x,y,z = [],[],[]
    with open(fn,'rb') as f:
        while True:
            try:
                s0 = f.read(1)
                # print(s0)
                if s0[0] == 0x7e:
                    s = f.read(8)
                    #print(s)
                    d = struct.unpack('!d',s)[0]
                    s = f.read(1)
                    if s[0] == 0x7d:
                        s = f.read(2)
                        ax = struct.unpack('!H',s)[0]
                        s = f.read(2)
                        ay = struct.unpack('!H',s)[0]
                        #print(s)
                        # print('ax={}'.format(ax))
                        # print('ay={}'.format(ay))
                        #
                        if d < 2060 or d> 2085:
                            continue

                        if ax >= 0xefff or ay >= 0xefff :
                            continue

                        x.append(ax)
                        y.append(ay)
                        z.append(d)
                    else:
                        continue
                else:
                    continue
            except:
                break
    return np.array(x),np.array(y),np.array(z)

def plot_xyz(x, y, z):
    # name = "TestData"
    marker = dict(size=2,line=dict(color=max(z)-z,width=0.5),
        opacity=0.6,
        colorscale='Viridis',
        color = z,
        showscale=True)
    trace=go.Scatter3d(x=x,y=y,z=z,mode='markers',marker=marker,name='markers')

    data = [trace]
    layout = go.Layout(margin=dict(l=0,r=0,b=0,t=0))
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='{}.html'.format(name))
    
def counting_filter(data):
    pcd = xyz2pcd(data)
    print("->正在进行统计滤波...")
    num_neighbors = 50 # K邻域点的个数
    std_ratio = 2 # 标准差乘数
    # 执行统计滤波，返回滤波后的点云sor_pcd和对应的索引ind
    sor_pcd, ind = pcd.remove_statistical_outlier(num_neighbors, std_ratio)
    sor_pcd.paint_uniform_color([0, 0, 1])
    print("valid:", sor_pcd)
    sor_pcd.paint_uniform_color([0, 0, 1])
    # 提取噪声点云
    sor_noise_pcd = pcd.select_by_index(ind,invert = True)
    # print("invalid:", sor_noise_pcd)
    sor_noise_pcd.paint_uniform_color([1, 0, 0])
    # ===========================================================

    #可视化统计滤波后的点云和噪声点云
    # o3d.visualization.draw_geometries[sor_pcd]
    # o3d.visualization.draw_geometries([sor_pcd, sor_noise_pcd])
    # o3d.visualization.draw_geometries([sor_pcd])
    return pcd2xyz(sor_pcd)
    
def radius_filter(data):
    pcd = xyz2pcd(data)
    print("->正在进行半径滤波...")
    # 执行半径滤波，返回滤波后的点云sor_pcd和对应的索引ind
    num_points = 10  # 邻域球内的最少点数，低于该值的点为噪声点
    radius = 50   # 邻域半径大小
    sor_pcd, ind = pcd.remove_radius_outlier(num_points, radius)
    sor_pcd.paint_uniform_color([0, 0, 1])
    print("valid:", sor_pcd)
    sor_pcd.paint_uniform_color([0, 0, 1])
    
    # 提取噪声点云
    sor_noise_pcd = pcd.select_by_index(ind,invert = True)
    print("invalid:", sor_noise_pcd)
    sor_noise_pcd.paint_uniform_color([1, 0, 0])
    
    # 可视化半径滤波后的点云和噪声点云
    
    # o3d.visualization.draw_geometries([sor_pcd])
    # o3d.visualization.draw_geometries([sor_pcd, sor_noise_pcd])
    # o3d.visualization.draw_geometries([sor_pcd])
    return pcd2xyz(sor_pcd)

def layers(data):
    # using _dbscan() method clustering the point cloud 
    # also can: kmeans
    
    # dbscan: eps = minimum distance between clusters; min_points = minimum numbers of clusters
    # points = pcd.points
    # points[:][2] = points[:][2]*1000000
    # pcd.points = points
    
    pcd = xyz2pcd(data)
    eps = 5000
    min_points = 10
    vectors = pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    labels = np.array(vectors)

    max_label = labels.max()
    print(f"Point Cloud has {max_label + 1} clusters")
    
    # o3d.visualization.draw_geometries(pcd, labels)
    return max_label, labels

def store_dic(dic, k, labels, points):
    # points = np.asarray(pcd.points)
    num = 0
    for i in range(k+1):
        layer = []
        nums = 0
        for j in range(len(labels)):
            # kk = [points[j][0], points[j][1], points[j][2]]
            if labels[j] == i:
                layer.append([points[j][0], points[j][1], points[j][2]])
                nums += 1
        layer = np.asarray(layer)
        # 去除特别小的集合，把误差产生的杂散点拍平
        if nums > 1000:
            dic.update({num:layer})
            num += 1
            print("Target Cluster Number {} has {} points, Z average is {}.".format(i, nums, sum(layer[:,2])/nums))
            ave = sum(layer[:,2])/nums
            # layer[:,2] = ave
            # layer[:,2] = 0.01*(layer[:,2] - ave) + ave
            # layer[:,2] = layer[:,2] + ave
            # plot_xyz(layer[:,0], layer[:,1], layer[:,2])
    return num, dic

def store_dic_2(dic, k, labels, points):
    # points = np.asarray(pcd.points)
    num = 0
    for i in range(k+1):
        layer = []
        nums = 0
        Maxn, Minn = 206000000, 2085000000
        Number = False
        for j in range(len(labels)):
            # kk = [points[j][0], points[j][1], points[j][2]]
            if labels[j] == i:
                layer.append([points[j][0], points[j][1], points[j][2]])
                nums += 1
            if points[j][2] > Maxn:
                Maxn = points[j][2]
            if points[j][2] < Minn:
                Minn = points[j][2]

        layer = np.asarray(layer)
        # 去除特别小的集合，把误差产生的杂散点拍平
        mmm = Maxn-Minn
        if  mmm< 2000000:
            Number = True
            
        if nums > 1000 and Number:
            dic.update({num:layer})
            num += 1
            print("Target Cluster Number {} has {} points, Z average is {}.".format(i, nums, sum(layer[:,2])/nums))
            ave = sum(layer[:,2])/nums
            # layer[:,2] = ave
            layer[:,2] = 0.01*(layer[:,2] - ave) + ave
            # layer[:,2] = layer[:,2] + ave
            plot_xyz(layer[:,0], layer[:,1], layer[:,2])
    return num, dic

def find_max_min(dic, dis, background, k):
    max_dis, min_dis = background, background+1
    if k < 3:
        return background
    
    if k == 3:
        for i in range(k):
            if dis[i] > dis[background]:
                max_dis = i
        return max_dis, background

    for i in range(k):
        if i == background:
            continue
        if dis[i] > background:
            if dis[i] > max_dis:
                max_dis = i
            if dis[i] < min_dis:
                min_dis = i
                
    return max_dis, min_dis

def remove_max_min(dic, max_dis, min_dis, background, k):
    points = np.asarray([])
    for i in range(k):
        if i == max_dis or i == min_dis or i == background:
            continue
        points.append(dic[i])
    return points

def load_args():
    argsfile = "args.txt"
    
    return 0


def dist(p1, p2, point):
    k = (p2[1]-p1[1]) / (p2[0]-p1[0])
    b = p1[1] - k*p1[0]
    dist = abs(k*point[0] - point[1] + b)/sqrt(1+b*b)
    return dist

def function2(fn):

    x, y, z = load_dat(fn)
    # plot_xyz(x, y, z)
    data = np.asarray([x,y,z]).T
    # print("File contains {} points".format(len(z)))
    x, y, z = [], [], []
    
    # y 取 2000-10000, x 取 0-7000
    # for i in data:
        
    #     if i[0] >7000 or i[1] > 10000:
    #         continue
    #     else:
    #         x.append(i[0])
    #         y.append(i[1])
    #         z.append(i[2])
    # data = np.asarray([x,y,z]).T
        
    # print("File contains {} points".format(len(z)))
    # plot_xyz(data[:,0], data[:,1], data[:,2])

    # 2 filter functions
    # data = radius_filter(data)
    data = counting_filter(data)

    # plot_xyz(data[:,0], data[:,1], data[:,2])
    
    k, labels = layers(data)

    # store layers in a dictionary
    dic = {}
    k, dic = store_dic(dic, k, labels, data)
    x,y,z = [],[],[]
    ave_1 = 0
    
    plane1 = dic[2]
    for i in plane1:
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])
        ave_1 += i[2]
    ave_1 = ave_1/len(z)
    
    plane2 = dic[3]
    for i in plane2:
        x.append(i[0])
        y.append(i[1])
        z.append(ave_1)
        
    # plane4 = dic[7]
    # for i in plane4:
    #     x.append(i[0])
    #     y.append(i[1])
    #     z.append(ave_1)
        
####################
   
    plane3 = dic[1]
    for i in plane3:
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])
        
    # plane4 = dic[4]
    # for i in plane4:
    #     x.append(i[0])
    #     y.append(i[1])
    #     z.append(i[2]-400000)
        
    # plane5 = dic[5]
    # for i in plane5:
    #     x.append(i[0])
    #     y.append(i[1])
    #     z.append(i[2]-300000)
        
    plot_xyz(x,y,z)
    
    with open(name+'.txt', 'w') as t:
        np.savetxt(t, np.asarray([x,y,z]).T)
    
'''
    # 取最下面一层
    zmin = min(data[:,2])
    x, y, z = [], [], []
    x2,y2,z2 = [],[],[]
    for i in data:
        if i[2] < zmin+0.5:
            x.append(i[0])
            y.append(i[1])
            z.append(i[2])
        elif i[2] < zmin+1.5:
            x2.append(i[0])
            y2.append(i[1])
            z2.append(zmin)
            
    # 缝隙，需要手动标定点
    # plot_xyz(x, y, z)
    # plot_xyz(x2, y2, z2)
    print("Size is {} points".format(len(z)))
    data = np.asarray([x,y,z]).T
    data2 = np.asarray([x2,y2,z2]).T
    k, labels = layers(data)
    
    dic = {}
    k, dic = store_dic(dic, k, labels, data)
    
    x,y,z = [],[],[]
    # 0.05
    l0p1 = [1360,7859]
    l0p2 = [1286,7565]
    l0q1 = [1646,7509]
    l0q2 = [1827,8015]
    for i in dic[0]:
        dis1 = dist(l0p1, l0p2, i) > 0.05
        dis2 = dist(l0q1, l0q2, i) > 0.05
        if dis1 and dis2:
            x.append(i[0])
            y.append(i[1])
            z.append(i[2]/10000)
            # z.append(zmin)
    # 0.1
    l1p1 = [4097,7421]
    l1p2 = [5231,7407]
    l1q1 = [3971,7214]
    l1q2 = [4840,6879]
    for i in dic[1]:
        dis1 = dist(l1p1, l1p2, i) > 0.01
        dis2 = dist(l1q1, l1q2, i) > 0.01
        if dis1 and dis2:
            x.append(i[0])
            y.append(i[1]) 
            z.append(i[2]/10000)
            # z.append(zmin)
    
    # plot_xyz(x,y,z)
    # 加入其他靶标上的点
    for i in data2:
        x.append(i[0])
        y.append(i[1])
        z.append(zmin*10)
        
    # data = np.asarray([x,y,z]).T
    # with open(fn+'.txt', 'w') as t:
    #     np.savetxt(t, np.asarray([x,y,z]).T)
    # plot_xyz(x,y,z)
'''

file_name = "0.05deg/fen_10.DAT"
name,pre = file_name.split('.DAT')
function2(file_name)

'''
读取使用np.loadtxt
    file_name = ""
    with open(file_name, 'r') as tt:
        dat = np.loadtxt(tt)
        x,y,z=[],[],[]
        for i in data2:
            x.append(i[0])
            y.append(i[1])
            z.append(zmin*10)
        plot_xyz(x,y,z)
'''

# name,pre = file_name.split('.DAT')

# x, y, z = load_dat(file_name)
# print("File contains {} points".format(len(z)))
# data = np.asarray([x,y,z]).T

# plot_xyz(data[:,0], data[:,1], data[:,2])

#     # 2 filter functions
#     # data = radius_filter(data)
# data = counting_filter(data)
# plot_xyz(data[:,0], data[:,1], data[:,2])