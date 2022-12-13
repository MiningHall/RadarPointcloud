import plotly
import plotly.graph_objs as go
import numpy as np
from numpy import sin,cos,mean,angle,pi,sinc,std,savetxt,rad2deg,arccos,sqrt, var
import matplotlib.pyplot as plt
import sys
import struct
import open3d as o3d

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
    name = "TestData"
    marker = dict(size=2,line=dict(color=max(z)-z,width=0.5),opacity=0.6,
        colorscale='Viridis',
        color = z,
        showscale=True)
    trace=go.Scatter3d(x=x,y=y,z=z,mode='markers',marker=marker,name='markers')

    data = [trace]
    layout = go.Layout(margin=dict(l=0,r=0,b=0,t=0))
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='xyz_{}.html'.format(name))
    
def counting_filter(data):
    pcd = xyz2pcd(data)
    print("->正在进行统计滤波...")
    num_neighbors = 100 # K邻域点的个数
    std_ratio = 5 # 标准差乘数
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
    num_points = 100  # 邻域球内的最少点数，低于该值的点为噪声点
    radius = 1000   # 邻域半径大小
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
    eps = 1000
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

def main_function(fn = "2022.11.14.5cm.DAT"):
    
    # load_args()
    # fn = "2022.11.14.5cm.DAT"
    name,pre = fn.split('.DAT')

    x, y, z = load_dat(fn)
    print("File contains {} points".format(len(z)))
    data = np.asarray([x,y,z]).T

    plot_xyz(data[:,0], data[:,1], data[:,2])

    # 2 filter functions
    # data = radius_filter(data)
    data = counting_filter(data)

    # get cluster numbers and label of points
    k, labels = layers(data)

    # store layers in a dictionary
    dic = {}
    k, dic = store_dic(dic, k, labels, data)

    # find target plane by Var()
    vars = {}
    dis = {}

    # find background from layers
    background = 0
    for i in range(k):
        print(i)
        layer = np.asarray(dic[i])
        std_z = np.std(layer[:,2])/len(layer[:,2])
        vars.update({i:std_z})
        if i > 0 and std_z < vars[background]:
            background = i
        dis_z = layer[:,2].sum()/ len(layer[:,2])
        dis.update({i:dis_z})
    plane1 = dic[background]
    # del dic[background]

    # find target from layers
    tar1 = background

    # if target exists front of backgroud
    for i in range(k):
        if dis[i] < dis[tar1]:
            tar1 = i
    plane2 = dic[tar1]
    
    # if target behind backgroud
    if tar1 == background:
        # find biggest and smallest cluster
        b, s = find_max_min(dic, dis, background, k)
        # remove biggest and smallest cluseter
        plane2 = remove_max_min(dic, b, s, background, k)
        
    plane2 = dic[k-1]
    
    x, y, z = [], [], []
    # for p in plane1:
    #     x.append(p[0])
    #     y.append(p[1])
    #     z.append(p[2]/1000)
        
    # for p in plane2:
    #     x.append(p[0])
    #     y.append(p[1])
    #     z.append(p[2]/1000)
        
    for i in range(k):
        planes = dic[i]
        for p in planes:
            x.append(p[0])
            y.append(p[1])
            z.append(p[2]/1000)
    # return plane2
    plot_xyz(x, y, z)

# file_name = "2022.11.14.5cm.DAT"
file_name = "2022.11.11-danban-0-10-ok.DAT"
main_function(file_name)

# if __name__ == "main":
#     file_name = "2022.11.14.5cm.DAT"
    # file_name = "2022.11.11-danban-0-10-ok.DAT"
    