import tkinter as tk
from os import system
try:
    from pickle import load
    from cv2 import imread,resize
    from PIL.ImageGrab import grab
except Exception:
    ok = system('pip install scikit-learn,torch,numpy,pandas,matplotlib,pillow,opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple')
    print('成功安装所需包...')if ok else print('未知错误...')

#事件_默认文本
def info_default(event):info['text'] = '无动作'
#事件_画图
def draw_track(event):
    l_canvas.create_oval(event.x,event.y,event.x+15,event.y+15,fill='white',outline='white',tag='track')
    info['text'] = f"坐标[{event.x},{event.y}]"
#事件_清除图像
def clear_track():l_canvas.delete('track')
#单选按钮选择事件
def selector():
    global model
    if flag.get() == 0:pass
    elif flag.get() == 1:
        with open('./models/clf_lr.pkl','rb')as f:model = load(f)
        info['text'] = 'LR模型加载完成!'
    elif flag.get() == 2:
        with open('./models/knc.pkl','rb')as f:model = load(f)
        info['text'] = 'KNN模型加载完成!'
    elif flag.get() == 3:
        with open('./models/cnn.pkl','rb')as f:model = load(f)
        info['text'] = 'CNN模型加载完成!'
#事件_执行分析
def execution():
    im=grab(bbox=(3,48,240,317))
    im.save('temp.png','PNG')
    gray = imread('./temp.png',0)
    flattened = resize(gray,(14,14)).reshape(1,-1)
    if flag.get() == 1 : info['text'] = f'LR预测: {model.predict(flattened)[0]}'
    elif flag.get() == 2 : info['text'] = f'KNN预测: {model.predict(flattened/255)[0]}'
    elif flag.get() == 3 : info['text'] = f'CNN预测: {model.predict(gray)[0]}'

model = None
selection = [
    ("LR模型",1),
    ("KNN模型",2),
    ("CNN模型",3)
]

main=tk.Tk()
main.title('MNIST手写识别')
main.geometry('480x320+0+0')
main.resizable(1,1)
flag = tk.IntVar()
#主体FRAME
frame=tk.Frame(main,width='480',height='320')
#标题label放在最上面
title_label=tk.Label(frame,width='480',height='1',text='MNIST手写识别',font=('Anrial',30),fg='yellow',bg='purple').pack(side='top')
#左边canvas画布
l_canvas=tk.Canvas(frame,width='240',height='320',bg='black',cursor='circle')
l_canvas.bind('<ButtonRelease-1>',info_default)
l_canvas.bind('<B1-Motion>',draw_track)
l_canvas.pack(side='left')
#右边FRAME
r_frame=tk.Frame(frame,width='240',height='320',bg='white')
r_frame.pack(side='right')
for t,n in selection:
    tk.Radiobutton(r_frame,width='240',height='1',text=t,value=n,variable=flag,command=selector,font=('Anrial',16,'bold'),relief='ridge',justify='left').pack(side='top')
#显示信息的messagebos
info=tk.Label(r_frame,width='240',height='6',text='无动作',font=('Anrial',16,'bold'),bg='gray',fg='red')
info.pack(side='top')
#右下层FRAME
rb_frame=tk.Frame(r_frame,width='240',height='60',bg='yellow')
#清除所画图形的按键
btn_clear=tk.Button(rb_frame,width='6',height='3',font=('Anrial',16,'bold'),fg='white',bg='yellow',text='清除',relief='ridge',command=clear_track).pack(side='left')
#执行分析的按键
btn_exec=tk.Button(rb_frame,width='6',height='3',font=('Anrial',16,'bold'),fg='white',bg='green',text='分析',relief='raised',command=execution).pack(side='left')
#退出的按键
btn_save=tk.Button(rb_frame,width='6',height='3',font=('Anrial',16,'bold'),fg='white',bg='red',text='退出',relief='solid',command=main.destroy).pack(side='left')

rb_frame.pack(side='bottom')
frame.pack()

main.overrideredirect(True)
main.mainloop()
