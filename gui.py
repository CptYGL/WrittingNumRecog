# 展示用,想用请执行 gui.py
import tkinter as tk
from os import system
try:
    from pickle import load
    from cv2 import imread,resize
    from PIL.ImageGrab import grab
    import torch
except Exception:
    ok = system('pip install torch scikit-learn numpy pandas matplotlib pillow opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple')
    print('成功安装所需包...')if ok else print('未知错误...')
    from pickle import load
    from cv2 import imread,resize
    from PIL.ImageGrab import grab
    import torch

#定义网络模型class Net(torch.nn.Module):
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2) # 卷积层,输入通道1,输出通道32,核大小5,步长1,填充2
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2) # 卷积层,输入通道32,输出通道64,核大小5,步长1,填充2
        self.bn1 = torch.nn.BatchNorm2d(64) # 归一化层
        self.pool = torch.nn.MaxPool2d(2,2) # 池化层,池化核大小2X2,步长2
        self.fc1 = torch.nn.Linear(7*7*64, 128) # 全连接层,输出节点128
        self.fc2 = torch.nn.Linear(128, 10) # 全连接层,输出节点10
    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.bn1(self.conv2(torch.nn.functional.relu(self.conv1(x))))))
        x = x.view(-1, 7*7*64)    # 把二维的数据展平成一维的
        x = torch.nn.functional.relu(self.fc1(x)) # 全连接层1
        x = self.fc2(x) # 全连接层2
        return x
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
        model = Net().cpu()
        model.load_state_dict(torch.load('./models/cnn_mnist.pkl',map_location=torch.device('cpu')))
        model.eval()
        info['text'] = 'CNN模型加载完成!'
#事件_执行分析
def execution():
    im=grab(bbox=(3,48,240,317))
    im.save('temp.png','PNG')
    gray = imread('./temp.png',0)
    gray = resize(gray,(14,14))
    flattened = gray.reshape(1,-1)
    if flag.get() == 1 : info['text'] = f'LR预测: {model.predict(flattened)[0]}'
    elif flag.get() == 2 : info['text'] = f'KNN预测: {model.predict(flattened/255)[0]}'
    elif flag.get() == 3 :
        img = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor)
        pred = torch.nn.functional.softmax(model(img),1)
        info['text'] = f'CNN预测: {max(pred)}'

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


# 转存为.py放置在RaspberryPi中