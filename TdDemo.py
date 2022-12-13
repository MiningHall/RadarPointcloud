
#-*- coding : utf-8-*-
# coding:unicode_escape
from PyQt5.QtCore import *
from PyQt5 import QtWidgets
from PIL import Image, ImageQt, ImageEnhance
from PyQt5 import QtGui
from PyQt5.QtWebEngineWidgets import *  # 导入浏览器的包
from PyQt5.QtWidgets import *
from collections import deque
from threading import Thread
from cv2 import Formatter_FMT_C
from numpy.core.fromnumeric import size
from numpy.lib.twodim_base import triu_indices_from
from numpy import deg2rad, rad2deg,arctan
from plotly.io import to_html
import plotly.graph_objs as go
import numpy as np
from pyqtgraph import functions, image
from pyqtgraph.metaarray.MetaArray import axis
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import gxipy as gx
import time, sys, tempfile, socket, upper, cv2, test4_copy
# import scipy

q = deque(maxlen=300000) # FIFO
counter = 0
temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False)
FOv = np.array([])
IMgae = np.array([])
pixlen = 0.0002 #单位像素距离

class Fov():
    def __init__(self):
        self.fov = np.array([])

    def Fov(self):
        global FOv, IMgae
        res = [] # 寻找连通域
        four_dot = [] #存放四个顶点索引
        d = 0.38
        pixlen = 0.0002  # 单位像素距离
        distance = 0.135 # 间距点实际距离
        # global numpyImage
        Width_set = 2560  # 设置分辨率宽
        Height_set = 2048  # 设置分辨率高
        framerate_set = 80  # 设置帧率
        exposuretime = 100000 # 设置曝光时间
        num = 1 # 采集帧率次数（为调试用，可把后边的图像采集设置成while循环，进行无限制循环采集）

        # 创建设备
        device_manager = gx.DeviceManager()  # 创建设备对象
        dev_num, dev_info_list = device_manager.update_device_list()  # 枚举设备，即枚举所有可用的设备


        # # 通过ip地址打开设备
        str_ip= dev_info_list[0].get("ip")
        cam = device_manager.open_device_by_ip(str_ip)

        # 设置宽和高
        cam.Width.set(Width_set)
        cam.Height.set(Height_set)
        # 设置连续采集
        # cam.TriggerMode.set(gx.GxSwitchEntry.OFF) # 设置触发模式
        cam.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)
        # 获取曝光值可设置范围和最大值
        float_range = cam.ExposureTime.get_range()
        float_max = float_range["max"]
        # 设置当前曝光值范围内任意值
        cam.ExposureTime.set(exposuretime)
        # # 获取当前曝光值
        # float_exposure_value = cam.ExposureTime.get()
        # 设置帧率
        cam.AcquisitionFrameRate.set(framerate_set)
        print("")
        print("**********************************************************")
        print("用户设置的帧率为:%d fps" % framerate_set)
        framerate_get = cam.CurrentAcquisitionFrameRate.get()  # 获取当前采集的帧率
        print("当前采集的帧率为:%d fps" % framerate_get)

        # 开始数据采集
        print("")
        print("**********************************************************")
        print("开始数据采集......")
        print("")
        cam.stream_on()

        # 采集图像
        for i in range(num):
            raw_image = cam.data_stream[0].get_image()  # 打开第0通道数据流
            if raw_image is None:
                print("获取彩色原始图像失败.")
                continue

            rgb_image = raw_image.convert("RGB")  # 从彩色原始图像获取RGB图像
            if rgb_image is None:
                continue

            # rgb_image.image_improvement(color_correction_param, contrast_lut, gamma_lut)  # 实现图像增强

            numpy_image = rgb_image.get_numpy_array()
            # numpyImage = numpy_image  # 从RGB图像数据创建numpy数组
            if numpy_image is None:
                continue
            # 原图
            # global img 
            img= Image.fromarray(numpy_image, 'RGB')  # 展示获取的图像


            # img.show()
            img.save("a6.jpg")
            # print('保存图像成功')
            # 把原图从PIL Image转成opencv Image
            contrast_enhancer = ImageEnhance.Contrast(img)
            img_enhanced_image = contrast_enhancer.enhance(2)
            enhanced_image = np.asarray(img_enhanced_image)
            r, g, b = cv2.split(enhanced_image)
            enhanced_image = cv2.merge([b, g, r])
        # # 停止采集
        print("")
        print("**********************************************************")
        print("摄像机已经停止采集")
        cam.stream_off()

        # 关闭设备
        print("")
        print("**********************************************************")
        print("系统提示您：设备已经关闭！")
        cam.close_device()

        def imgfov(img2):
            
            img4 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            # 高斯去噪
            blur = cv2.GaussianBlur(img4, (5, 5), 0)
            edges = cv2.Canny(blur, 145, 35)
            # #定义结构元素/设置卷积核
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            # 先开运算再闭运算
            opened = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
            #一定要先开闭运算再进行二值化！！
            ret3, img3 = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # circles = cv2.HoughCircles(img3, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
            # circles = np.uint16(np.around(circles))  # 把circles包含的圆心和半径的值变成整数
            contours1, hierarchy = cv2.findContours(img3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours1[0]
            # print('cnt的长度为',len(cnt)) 
            # dot1_cnt = contours[len(contours)-1]
            # dot2_cnt = contours[len(contours)-2]
            cv2.drawContours(img2, contours1, 0, (0,255,0),10)

            # cv2.imshow('',addimg)
            # 初始X和Y的最大最小值
            Xmax = cnt[0][0][0]
            Ymax = cnt[0][0][1]
            Xmin = cnt[0][0][0]
            Ymin = cnt[0][0][1]
            # print(Xmax,Ymax,Xmin,Ymin)
            # 遍历找到X坐标最大，Y坐标最大最小的索引
            for i in range(len(cnt)):
                if cnt[i][0][0] > Xmax:
                    Xmax = cnt[i][0][0]
                    index_Xmax = i
                if  cnt[i][0][1] > Ymax:
                    Ymax = cnt[i][0][1]
                    index_Ymax = i
            for j in range(len(cnt)):
                if cnt[j][0][0] < Xmin:
                    Xmin = cnt[j][0][0]
                    index_Xmin = j
                if cnt[j][0][1] <= Ymin:
                    Ymin = cnt[j][0][1]
                    index_Ymin = j
            topleft = contours1[0][index_Xmin][0]
            topright = contours1[0][index_Ymin][0]
            bottomleft = contours1[0][index_Xmax][0]
            bottomright = contours1[0][index_Ymax][0]
            print('四个顶点坐标为：',topleft,topright,bottomleft,bottomright)
            hor_x = max(topleft[0],topright[0],bottomleft[0],bottomright[0])-min(topleft[0],topright[0],bottomleft[0],bottomright[0])
            hor_y = max(topleft[1],topright[1],bottomleft[1],bottomright[1])-min(topleft[1],topright[1],bottomleft[1],bottomright[1])
            print('最大x',hor_x)
            print('最大y',hor_y)
            # # 水平实际距离
            realityHorizontal = hor_x * pixlen
            # # 垂直实际距离
            realityVertical = hor_y * pixlen
            FOVh = 1.2*2*rad2deg(arctan(realityHorizontal / d / 2))
            FOVv = 2*rad2deg(arctan(realityVertical / d / 2))
            print('水平扫描范围和垂直扫描范围分别为：',FOVh,FOVv)
            print(realityHorizontal,realityVertical)
            # # 计算实际长度和宽度
            return FOVh, FOVv
        # img5 = cv2.imread('./imge/20.bmp',0)
        Fov_h, Fov_v = imgfov(enhanced_image)
        FOv = [Fov_h, Fov_v]
        IMgae = numpy_image

    def save(self):

        res = [] # 寻找连通域
        four_dot = [] #存放四个顶点索引
        d = 0.38
        pixlen = 0.0002  # 单位像素距离
        distance = 0.135 # 间距点实际距离
        # global numpyImage
        Width_set = 2560  # 设置分辨率宽
        Height_set = 2048  # 设置分辨率高
        framerate_set = 80  # 设置帧率
        exposuretime = 100000 # 设置曝光时间
        num = 1 # 采集帧率次数（为调试用，可把后边的图像采集设置成while循环，进行无限制循环采集）

        # 创建设备
        device_manager = gx.DeviceManager()  # 创建设备对象
        dev_num, dev_info_list = device_manager.update_device_list()  # 枚举设备，即枚举所有可用的设备


        # # 通过ip地址打开设备
        str_ip= dev_info_list[0].get("ip")
        cam = device_manager.open_device_by_ip(str_ip)

        # 设置宽和高
        cam.Width.set(Width_set)
        cam.Height.set(Height_set)
        # 设置连续采集
        # cam.TriggerMode.set(gx.GxSwitchEntry.OFF) # 设置触发模式
        cam.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)
        # 获取曝光值可设置范围和最大值
        float_range = cam.ExposureTime.get_range()
        float_max = float_range["max"]
        # 设置当前曝光值范围内任意值
        cam.ExposureTime.set(exposuretime)
        # # 获取当前曝光值
        # float_exposure_value = cam.ExposureTime.get()
        # 设置帧率
        cam.AcquisitionFrameRate.set(framerate_set)
        print("")
        print("**********************************************************")
        print("用户设置的帧率为:%d fps" % framerate_set)
        framerate_get = cam.CurrentAcquisitionFrameRate.get()  # 获取当前采集的帧率
        print("当前采集的帧率为:%d fps" % framerate_get)

        # 开始数据采集
        print("")
        print("**********************************************************")
        print("开始数据采集......")
        print("")
        cam.stream_on()

        # 采集图像
        for i in range(num):
            raw_image = cam.data_stream[0].get_image()  # 打开第0通道数据流
            if raw_image is None:
                print("获取彩色原始图像失败.")
                continue

            rgb_image = raw_image.convert("RGB")  # 从彩色原始图像获取RGB图像
            if rgb_image is None:
                continue

            # rgb_image.image_improvement(color_correction_param, contrast_lut, gamma_lut)  # 实现图像增强

            numpy_image = rgb_image.get_numpy_array()
            # numpyImage = numpy_image  # 从RGB图像数据创建numpy数组
            if numpy_image is None:
                continue
            # 原图
            # global img 
            img= Image.fromarray(numpy_image, 'RGB')  # 展示获取的图像


            # img.show()
            img.save("a6.jpg")
            # print('保存图像成功')
            # 把原图从PIL Image转成opencv Image
            contrast_enhancer = ImageEnhance.Contrast(img)
            img_enhanced_image = contrast_enhancer.enhance(2)
            enhanced_image = np.asarray(img_enhanced_image)
            r, g, b = cv2.split(enhanced_image)
            enhanced_image = cv2.merge([b, g, r])
        # # 停止采集
        print("")
        print("**********************************************************")
        print("摄像机已经停止采集")
        cam.stream_off()

        # 关闭设备
        print("")
        print("**********************************************************")
        print("系统提示您：设备已经关闭！")
        cam.close_device()
        
class funCollection():
    def __init__(self, ui):
        self.ui = ui
        self.distData = np.empty(3000)

    def saveImage(self):
        fov = Fov()
        fov.save()
        self.showFov()

    def showFov(self):
        test4_copy.fov()
        # Img = Image.fromarray(test4_copy.image.astype(np.uint8))
        # pixImg = ImageQt.toqpixmap('a6.jpg')
        

        # 显示图片
        self.ui.label_3.setPixmap(ImageQt.QPixmap('a6.jpg'))
        self.ui.label_3.setScaledContents(True)


        # imageSize = self.ui.label_3.size()
        # image=QtGui.QImage(Fov.img)
        # image = QtGui.QPixmap(Fov.img).scaled(imageSize)
        # self.ui.Label_3.setPixmap(image)

        self.ui.lineEdit_4.setText(str(1.2*test4_copy.Fov_h))
        self.ui.lineEdit_5.setText(str(test4_copy.Fov_v))
        # self.ui.lineEdit_6.setText(str(pixlen))

    def getDis(self):
        # 这个地方的 plot 要加在哪里还需要好好设计一下
        DisPlot = self.ui.addPlot()
        DisCurve = DisPlot.plot(self.distData)
        while(True):
            Dis = q.pop()
            q.append(Dis)
            self.ui.lineEdit.setText(str(Dis[2]))
            self.updateDisatance(Dis)
            DisCurve.setData(self.distData)
            time.sleep(0.05)
            
    def updateDisatance(self, Dis):
        # 前提是已经有了一个窗口，再调用这个函数实时更新数据
        if np.size(self.distData) <= 3000:
            np.append(self.distData, Dis)
        else:
            self.distData[:-1] = self.distData[1:]
            self.distData[-1] = Dis
        
    def showDis(self):
            t3 = Thread(target=self.getDis, args=())
            Thread.start(t3)

    def average_5(self):
        s = int(np.size(q)/3)
        Axis = np.array(q)
        # print(type(Axis))
        sum = Axis[s-1,2]+Axis[s-2,2]+Axis[s-3,2]+Axis[s-4,2]+Axis[s-5,2]
        # print(sum)
        self.ui.lineEdit_2.setText(str(sum/5))

    def counter20(self):
        while(1):
            global counter
            self.ui.lineEdit_3.setText(str(counter)+'点/秒')
            counter = 0
            time.sleep(1)

    def FPS(self):
        print(counter)
        timer = pg.QtCore.QTimer()
        timer.timeout.connect(self.counter20)
        timer.start(10)
        

        

class Client():
    # 此处需要添加一个计数器，接收click信号清零，且每隔固定时间清零一次
    # 作为帧频率的统计计数，每秒点数大于 6e6 个
    def __init__(self, Ui, port=1235):
        self.s = socket.socket()  # 创建 socket 对象
        self.host = socket.gethostname() # 获取本地主机名
        # self.s.connect((self.host, port))
        self.fixnum = 300000
        self.ui = Ui
        # self.counter = 0
        # self.lock = threading.Lock()

    def rec(self):
        global counter
        global num
        str_bytes = self.s.recv(1000)
        # bytes -> int -> 0b -> xyzt
        int_str = int().from_bytes(str_bytes, byteorder='little', signed=True)
        # print(int_str)
        bin_str = bin(int_str).replace('0b', '')
        bin_str = '0'*(40-len(bin_str))+bin_str
        # print(len(bin_str))
        # print(bin_str)
        # print(bin_str[0:9], bin_str[9:19], bin_str[19:31], bin_str[31:40])
        x = int(bin_str[0:9], 2) - 2**8
        y = int(bin_str[9:19], 2) - 2**9
        z = int(bin_str[19:31], 2) - 2**11
        t = int(bin_str[31:40], 2) - 2**8
        # print(x,y,z,t)
        if z > 9:
            counter += 1
            q.append([float(x)/10, float(y)/20, float(z)/20])

    def filt(self, str_):
        if str_[0] != ',':
            str_ = str_.strip("\n").split(',')
            q.append([np.float32(str_[1]), np.float32(str_[3]), np.float32(str_[5])])

    def accept(self):
        while True:
            self.rec()

class show_window(QWebEngineView):
    def __init__(self, ui):
        super().__init__()
        self.ui = ui
        self.widget = gl.GLViewWidget()
        self.webView = QWebEngineView(self.widget)
        self.ui.verticalLayout_6.addWidget(self.widget)
        self.wSize = [self.widget.deviceWidth(), self.widget.deviceHeight()]
        self.webView.setGeometry(QRect(0, 0, self.wSize[0], self.wSize[1]))  # (起始点+页面长宽)
        self.freshPlot()

    # def setPicture(self, ui):
    #     Picture = QtGui.QPixmap(image).scaled(400, 400)
    #     ui.verticalLayout_9.setPixmat(Picture)

    def freshPlot(self):
        self.thread = PlotFig()
        self.thread.signal.connect(self.freshWebview)
        self.thread.start()

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.thread.run)
        self.timer.start(100000)

    def Resize(self):
        nSize = [self.widget.deviceWidth(), self.widget.deviceHeight()]
        if self.wSize != nSize:
            self.wSize = nSize
            self.webView.setGeometry(QRect(0, 0, self.wSize[0], self.wSize[1]))  # (起始点+页面长宽)
            self.ui.label_3.setMaximumSize(self.wSize[0], self.wSize[1])
    def freshWebview(self, Url):
        # print('freshWebview')
        self.webView.load(QUrl.fromLocalFile(temp_file.name))

class PlotFig(QThread):
    signal = pyqtSignal(str)
    def __init__(self):
        super(PlotFig, self).__init__()
        self.Axis = np.empty((0,3))

    def set_figure(self, fig=None):
        temp_file.seek(0) # 移动文件读取指针到指定位置
        fig.update_xaxes(showspikes=True)
        fig.update_yaxes(showspikes=True)
        html = to_html(fig, config={"responsive": True, 'scrollZoom': True})
        html += "\n<style>body{margin: 0;}" \
                "\n.plot-container,.main-svg,.svg-container{width:100% !important; height:100% !important;}</style>"

        temp_file.write(html)
        temp_file.truncate() # 从当前位置起截断，截断之后位置后的数据都为0
        temp_file.seek(0)
        print('Tempfile writed')

    def plotFig(self):
        x = self.Axis[:,0]
        y = self.Axis[:,1]
        z = self.Axis[:,2]
        fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.5)]) # 作图
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        # fig.show()
        self.set_figure(fig)
        
    def run(self):
        if q:
            self.Axis = np.append(self.Axis, q, axis=0)
            Axiss = np.empty((0,3))
            # Axiss[0,:] = self.Axis[0,:]
            for i in range(self.Axis.ndim):
                Axiss = np.append(Axiss, [self.Axis[i,:]], axis=0)
                Axiss = np.append(Axiss, (self.Axis[i,:]+self.Axis[i+1:])/2, axis=0)
            # Axiss = np.append(Axiss, [self.Axis[size(self.Axis)]], axis=0)
            self.Axis = Axiss
            self.plotFig()
            self.signal.emit(str(QUrl.fromLocalFile(temp_file.name)))
            self.Axis = np.empty((0,3))


def main():

    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = upper.Ui_MainWindow()

    Cl = Client(ui)
    t1 = Thread(target=Cl.accept, args=())
    Thread.start(t1)

    Functions = funCollection(ui)

    ui.setupUi(mainWindow)
    mainWindow.show()
    window = show_window(ui)

    timer = pg.QtCore.QTimer()
    timer.timeout.connect(window.Resize)
    timer.start(500)

    t2 = Thread(target=Functions.counter20, args=())
    Thread.start(t2)

    
    ui.pushButton.clicked.connect(window.freshPlot)
    ui.pushButton_2.clicked.connect(Functions.average_5)
    # ui.pushButton_3.clicked.connect(Functions.FPS)
    ui.pushButton_4.clicked.connect(Functions.saveImage)
    ui.pushButton_5.clicked.connect(Functions.saveImage)
    
    sys.exit(app.exec_())
if __name__ == '__main__':
    main()