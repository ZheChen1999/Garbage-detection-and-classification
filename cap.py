from tkinter import *
import cv2 as cv
from PIL import Image, ImageTk
import os
from detect import *

import argparse
import time
from sys import platform
from tkinter import *
from models import *
from utils.datasets import *
from utils.utils import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL.ImageTk import PhotoImage
import serial

test= ["待测样本"]

def uart(choice):
    t = serial.Serial('/dev/ttyUSB0',9600)
    if choice == 1:
        a = "\x01"
    elif choice == 2:
        a = "\x02"
    elif choice == 3:
        a = "\x03"
    elif choice == 5:
        a = "\x05"
    elif choice == 6:
        a = "\x06"
    elif choice == 7:
        a = "\x07"
    t.write(a.encode('utf-8'))

def jiaodu(num):
    if num == 1:                      # PWM_A  123
        uart(3)
        for i in range(4):
            uart(2)
        time.sleep(0.3)

        uart(7)
        for i in range(2):
            uart(6)
        for i in range(5):
            uart(5)
       
        time.sleep(1)

        uart(7)
        for i in range(2):
            uart(6)

    elif num == 2:
        uart(3)
        for i in range(2):
            uart(2)
        time.sleep(0.3)

        uart(7)
        for i in range(2):
            uart(6)
        for i in range(5):
            uart(5) 
       
        time.sleep(1)

        uart(7)
        for i in range(2):
            uart(6)

    elif num == 3:
        uart(3)
        for i in range(2):
            uart(1)
        time.sleep(0.3)

        uart(7)
        for i in range(2):
            uart(6)
        for i in range(5):
            uart(5) 
       
        time.sleep(1)

        uart(7)
        for i in range(2):
            uart(6)
    

    elif num == 4:
        uart(3)
        for i in range(4):
            uart(3)
        time.sleep(0.3)
        uart(7)
        for i in range(2):
            uart(6)
        for i in range(5):
            uart(5)
       
        time.sleep(1)

        uart(7)
        for i in range(2):
            uart(6)

def detect(cfg,
           data,
           weights,
           images='data/samples',  # input folder
           output='output',  # output folder
           fourcc='mp4v',  # video codec
           img_size=416,
           conf_thres=0.5,
           nms_thres=0.5,
           save_txt=False,
           save_images=True,
           webcam=False):
    # Initialize

    frame10 = Frame()
    frame10.pack()
                # 容器框 （LabelFrame）
    #frame10.group = LabelFrame(frame10, text="检测结果", padx=5, pady=5)
    #frame10.group.grid()

    device = torch_utils.select_device()
    torch.backends.cudnn.benchmark = True  # set False for reproducible results
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    if ONNX_EXPORT:
        s = (320, 192)  # (320, 192) or (416, 256) or (608, 352) onnx model image size (height, width)
        model = Darknet(cfg, s)
    else:
        model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3, s[0], s[1]))
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadImages(images, img_size=img_size)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    
    for i, (path, img, im0, vid_cap) in enumerate(dataloader):
        t = time.time()
        save_path = str(Path(output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        pred, _ = model(img)
        det = non_max_suppression(pred, conf_thres, nms_thres)[0]

        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results to screen
            print('%gx%g ' % img.shape[2:], end='')  # print image size
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                print('%g %ss' % (n, classes[int(c)]), end=', ')

            # Draw bounding boxes and labels of detections
            for *xyxy, conf, cls_conf, cls in det:
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))
                
                # Add bbox to the image
                label = '%s %.2f' % (classes[int(cls)], conf)
                jiaodu(int(cls))
                   
                #test.append(label)
                #w = Label(frame10.group, text=label)
                #w.pack()
                output=label
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
        speed='Done. (%.3fs)' % (time.time() - t)
        print('Done. (%.3fs)' % (time.time() - t))
        #w = Label(frame10.group, text=speed)
        #w.pack() 
        if webcam:  # Show live webcam
            cv2.imshow(weights, im0)
            
        if save_images:  # Save image with detections
            if dataloader.mode == 'images':
                cv2.imwrite(save_path, im0)
                imgcz = cv2.imread(save_path)

                
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
                vid_writer.write(im0)

    if save_images:
        print('Results saved to %s' % os.getcwd() + os.sep + output)
        if platform == 'darwin':  # macos
            os.system('open ' + output + ' ' + save_path)
#临时变量
tempimagepath_cam     = r"/home/cz/yolov3/data/cam/1.jpg"
tempimagepath_detect  = r"/home/cz/yolov3/data/samples/1.jpg"
path = "/home/cz/yolov3/output/1.jpg"

path_set= "/home/cz/yolov3/data_set/"
count=0
#摄像机设置
#0是代表摄像头编号，只有一个的话默认为0
capture=cv.VideoCapture(0) 

def getframe():
    ref,frame=capture.read()
    cv.imwrite(tempimagepath,frame)

def closecamera():
    capture.release()

#界面相关
window_width=1000
window_height=1000
image_width=int(window_width*0.4)
image_height=int(window_height*0.4)
imagepos_x=int(window_width*0.1)
imagepos_y=int(window_height*0.1)

image1_width=int(window_width*0.4)
image1_height=int(window_height*0.4)
imagepos1_x=int(window_width*0.1)
imagepos1_y=int(window_height*0.55)

global output

butpos_x=1500
butpos_y=500

butpos1_x=1500
butpos1_y=600

butpos2_x=900
butpos2_y=600

top=Tk()
top.wm_title("garbage classify of scuec")
top.geometry(str(window_width)+'x'+str(window_height))
image2 =Image.open(r'/home/cz/yolov3/label/scuec.png')
background_image = ImageTk.PhotoImage(image2)

#界面相关
window_width=1000
window_height=1000
top.geometry(str(window_width)+'x'+str(window_height))

background_label = Label(top, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)



icon = PhotoImage(file="/home/cz/yolov3/label/scuec.png")
top.call('wm', 'iconphoto', top._w, icon)




def usb_or_ip():
    l.config(text='you have selected'+var.get())
    return var.get()

def ip_image():
    video="http://" + get_ip() + "/video?dummy=param.mjpg"
    capture =cv.VideoCapture(video)
    success,frame = capture.read()

    cvimage = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
    pilImage=Image.fromarray(cvimage)
    pilImage = pilImage.resize((image_width, image_height),Image.ANTIALIAS)
    tkImage =  ImageTk.PhotoImage(image=pilImage)
    return tkImage

var=StringVar()

l=Label(top,bg='yellow',width=20,text='empty')
#l.place(x=10,y=100,anchor='nw')
l.pack()
#text参数为显示在选择按钮上为Option A, 当选择Option A的时候，把var赋值成value参数A，
r1=Radiobutton(top,text='Option A USB-CAMERA(default)',variable=var,value='A',command=usb_or_ip)
r1.pack()

r2=Radiobutton(top,text='Option B PHONE-IP-CAMERA',variable=var,value='B',command=usb_or_ip)
r2.pack()





#Entry(输入)
#第1步
e=Entry(top,show=None) 
#比如像密码那样输入：show='*'
#第2步，把Entry放在window上面
e.pack()

#第3步，定义button
def get_ip():
    var=e.get()
    return var
b1=Button(top,text="get IP-address",width=15,height=2,command=get_ip)
b1.pack()  #把button放在label下面的位置




def tkImage():
   choice=usb_or_ip()
   if choice == "B":
        video="http://" + get_ip() + "/video?dummy=param.mjpg"
        capture1 =cv.VideoCapture(video)
        success1,frame1 = capture1.read()
        cvimage = cv.cvtColor(frame1, cv.COLOR_BGR2RGBA)
        pilImage=Image.fromarray(cvimage)
        pilImage = pilImage.resize((image_width, image_height),Image.ANTIALIAS)
        tkImage =  ImageTk.PhotoImage(image=pilImage)
        return tkImage
   else:
        ref,frame=capture.read()
        cvimage = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
        pilImage=Image.fromarray(cvimage)
        pilImage = pilImage.resize((image_width, image_height),Image.ANTIALIAS)
        tkImage =  ImageTk.PhotoImage(image=pilImage)
        return tkImage


def button1():
    choice=usb_or_ip()
    global count
    if choice == "B":
        video="http://" + get_ip() + "/video?dummy=param.mjpg"
        capture1 =cv.VideoCapture(video)
        success1,frame1 = capture1.read()
        cv.imwrite(tempimagepath_cam,frame1)
        cv.imwrite(tempimagepath_detect,frame1[85:450, 200:480])
        cv.imwrite(path_set +"new"+ str(count)+".jpg",frame)
        count=count+1
    else:
        success,frame = capture.read()
        ref,frame=capture.read()
        cv.imwrite(tempimagepath_detect,frame[85:450, 200:480])
        cv.imwrite(tempimagepath_cam,frame)
        cv.imwrite(path_set +"new"+ str(count)+".jpg",frame)
        count=count+1
def button2():
    choice=usb_or_ip()
    if choice == "B":
        video="http://" + get_ip() + "/video?dummy=param.mjpg"
        capture1 =cv.VideoCapture(video)
        success1,frame1 = capture1.read()
        cv.imwrite(tempimagepath_cam,frame1)
        cv.imwrite(tempimagepath_detect,frame1[85:450, 200:480])
    else:
        success,frame = capture.read()
        ref,frame=capture.read()
        cv.imwrite(tempimagepath_detect,frame[85:450, 200:480])
        cv.imwrite(tempimagepath_cam,frame)
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='data/train/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='cz.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='data/yolov3_10000.weights', help='path to weights file')
    parser.add_argument('--images', type=str, default='data/samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='fourcc output video codec (verify ffmpeg support)')
    parser.add_argument('--output', type=str, default='output', help='specifies the output path for images and videos')
    opt = parser.parse_args()

    with torch.no_grad():
        label=detect(opt.cfg,
               opt.data,
               opt.weights,
               images=opt.images,
               img_size=opt.img_size,
               conf_thres=opt.conf_thres,
               nms_thres=opt.nms_thres,
               fourcc=opt.fourcc,
               output=opt.output)
                # frame10

#控件定义
canvas =Canvas(top,bg='white',width=image_width,height=image_height)#绘制画布

canvas.place(x=imagepos_x,y=imagepos_y)

canvas1 =Canvas(top,bg='white',width=image1_width,height=image1_height)#绘制画布

canvas1.place(x=imagepos1_x,y=imagepos1_y)

b=Button(top,text='save picture',width=15,height=2,command=button1)
b.place(x=butpos_x,y=butpos_y)

b1=Button(top,text='detect picture',width=15,height=2,command=button2)
b1.place(x=butpos1_x,y=butpos1_y)

predict = Canvas(
        top,
        width=400,
        height=400,
        background="white"
    )
predict.place(x=1450,y=50)



if __name__=="__main__":
   
   while(True):
     
     picture=tkImage()
     canvas.create_image(0,0,anchor='nw',image=picture)
     
     imgcz    =  cv.imread(path)
     cvimage  =  cv.cvtColor(imgcz, cv.COLOR_BGR2RGBA)
     pilImage =  Image.fromarray(cvimage)
     pilImage =  pilImage.resize((image_width, image_height),Image.ANTIALIAS)
     tkImage1 =  ImageTk.PhotoImage(image=pilImage)
     canvas1.create_image(0,0,anchor='nw',image=tkImage1)
     
     img1    =  cv.imread(tempimagepath_detect)
     cvimage  =  cv.cvtColor(img1, cv.COLOR_BGR2RGBA)
     pilImage =  Image.fromarray(cvimage)
     pilImage =  pilImage.resize((image_width, image_height),Image.ANTIALIAS)
     tkImage2 =  ImageTk.PhotoImage(image=pilImage)
     predict.create_image(0,0,anchor='nw',image=tkImage2)
     #time.sleep(0.5)
     #button2()
     top.update()
     top.after(10)     
   
   top.mainloop()  
   closecamera()

