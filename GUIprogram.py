'''Author @185150736YGL'''
#有关PIL-PythonImageLibrary和scikit-image和win32gui的使用从网上宕下来的
import sys
import tkinter as tk
import digitClassifier as dc
from PIL import ImageGrab,Image,ImageTk         #pip install pillow
from skimage import io
import win32gui

#print(knn.classifier('1.png'))

#事件_默认文本
def info_default(event):
    info['text']=defaultinfo
#事件_画图
def draw_track(event):
    l_canvas.create_oval(event.x,event.y,event.x + 15,event.y + 15,fill = 'white',outline = 'white',tag = 'track')
    info['text']=f"Drawing on X:{event.x} Y:{event.y}"
#事件_清除图像
def clear_track(event):
    l_canvas.delete('track')
    r_canvas.delete('text')
    r_canvas.delete('img')
#事件_保存图像
def save_file(event):
    HWND = win32gui.GetFocus()                  #来源网络
    rect=win32gui.GetWindowRect(HWND)           #来源网络
    im=ImageGrab.grab(rect)                     #来源网络
    im.save('temp.png','PNG')
    a=io.imread('temp.png')
    b=a[46:550,0:400]
    io.imsave('test.png',b)
    knn.trainKNN()
    knn.classifier('test.png')
    
#事件_执行分析
def knn_exec(event):
    f = open('similarity.log', mode = 'r',encoding='utf-8').readlines()
    r_canvas.create_text(270,220,text = f[0:21],tag = 'text')
    info['text'] = f[22:23]
    im = ImageTk.PhotoImage(Image.open('gray.png'))
    r_canvas.image = im
    r_canvas.create_image(5,5,anchor = 'nw',image = im,tag = 'img')
    

defaultinfo = 'NO MOVEMENT YET...'
knn=dc.DigitKNNClassifier()
#img = None
im = None

main = tk.Tk()
main.title('Computer Vision Interface')
main.geometry('1000x600')
main.resizable(0,0)
#主体FRAME
frame = tk.Frame(main,width = '800',height = '600')
#标题label放在最上面
title_label = tk.Label(frame,width = '800',height = '1',text = '-COMPUTER VISION RECOGNITION-',font=('Anrial',30),fg = 'white',bg = 'blue').pack(side = 'top')

#中间层FRAME
mid_frame = tk.Frame(frame,width = '800',height = '500',bg = 'blue')
#左边canvas画布
l_canvas = tk.Canvas(mid_frame,width = '450',height = '500',bg = 'black',cursor = 'circle')
l_canvas.bind('<ButtonRelease-1>',info_default)
l_canvas.bind('<B1-Motion>',draw_track)
l_canvas.pack(side = 'left')
r_canvas = tk.Canvas(mid_frame,width = '550',height = '500',bg = 'grey')
r_canvas.pack(side = 'right')

mid_frame.pack(side = 'top')

#最下层FRAME
end_frame = tk.Frame(frame,width = '800',height = '50',bg = 'yellow')
#清除所画图形的按键
btn_clear = tk.Button(end_frame,width = '8',height = '50',text = 'CLEAR',relief = 'ridge')
btn_clear.bind('<Button-1>',clear_track)
btn_clear.pack(side = 'left')
#保存图像的按键
btn_save = tk.Button(end_frame,width = '8',height = '50',text = 'SAVE',relief = 'solid')
btn_save.bind('<Button-1>',save_file)
btn_save.pack(side = 'left')
#执行分析的按键
btn_exec = tk.Button(end_frame,width = '8',height = '50',bg = 'pink',text = 'ANALYSIS',relief = 'raised')
btn_exec.bind('<Button-1>',knn_exec)
btn_exec.pack(side = 'left')
#显示信息的label
info = tk.Label(end_frame,width = '50',height = '50',text = defaultinfo,font = ('Anrial',22,'bold'),bg = 'green',fg = 'red')
info.pack(side = 'left')
end_frame.pack(side = 'bottom')


frame.pack()
main.mainloop()
