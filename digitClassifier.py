import numpy as np
import cv2
import sys
#定义类和方法
temp = sys.stdout
class DigitKNNClassifier(object):
    knn = cv2.ml.KNearest_create()
    
    def trainKNN(self):
        '''self,参数,类内方法'''
        img = cv2.imread('digits.png')
        '''cv2.imread方法,即image-read,读取图像(训练集)'''
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        '''cv2.cvtcolor方法,即convert-color,转换为灰度图'''
        cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
        '''
        numpy.hsplit: horizonal-split横轴拆分
        numpy.hsplit: vertical-split纵轴拆分
        先将灰度图纵分50,再横分100
        图片2000pix*1000pix,则cells为50行*100列的单体图片的集合[]
        '''
        x = np.array(cells)
        '''转成numpy数组，数组元素实际是20x20的小矩阵'''
        
        train = x[:,:50].reshape(-1,400).astype(np.float32)
        '''前50列用作训练'''
        test = x[:,50:100].reshape(-1,400).astype(np.float32)
        '''后50列用作测试'''
        '''astype(np.float32) == [array],dtype=float,转化为32bit浮点数'''

        #创建训练和测试的标签
        k = np.arange(10)
        '''0~9的10个数字'''
        train_labels = np.repeat(k,250)[:,np.newaxis]
        test_labels = np.repeat(k,250)[:,np.newaxis]
        '''
        arange类似于range，返回一个array对象
        repeat表示每个元素复制250次，这里的np.newaxis表示在列上复制
        每个数字5行50列
        测试集标签与训练集相同,都tag上对应数字
        '''
        
        knnc=self.knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)
        '''cv2.ml.ROW_SAMPLE，数组以行的方式布局，为了方便knn算法计算'''
        ret,result,neighbours,dist = self.knn.findNearest(test,k=5)         #用法???
        
        matches = result == test_labels
        '''使用测试集统计准确率'''
        correct = np.count_nonzero(matches)
        '''numpy.count_zero方法,取非零计数,给correct'''                     #result???
        accuracy = correct*100.0 / result.size
        '''结果为1(true)的总数/总数'''
        sys.stdout = open('similarity.log', mode = 'w',encoding='utf-8')
        print("测试集准确率：",accuracy)
        sys.stdout = temp

    def classifier(self,imgpath):
        '''self,参数,类内方法'''
        img = cv2.imread(imgpath)
        '''读取需要测试的图'''
        #imgp=cv2.imdecode(np.fromfile(imgpath,dtype=np.uint8),-1)           #解决中文路径 image-decode转码       
        #cv2.imshow('original',img)
        '''展示源图片'''
        img = cv2.resize(img,(20,20))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        '''调整图片尺寸，转为灰度图'''
        cv2.imwrite('./gray.png',gray)
        a = np.array(gray)
        '''把图片转成numpy数组,元素实际是20*20的小矩阵'''
        x = a.reshape((-1,400)).astype(np.float32)
        retval,result=self.knn.predict(x)
        '''retval,return-value,存返回值'''
        sys.stdout = open('similarity.log', mode = 'a+',encoding='utf-8')
        for i in range(20):
            for k in range(20):
                sys.stdout.write('{0: ^5}'.format(str(a[i][k])))
            sys.stdout.write('\n')
        print(result)
        print('你所输入的数字是 [ '+str(int((result[0][0])))+' ] 吗?')
        sys.stdout = temp
        '''返回result第一个'''


#knn = DigitKNNClassifier()
#knn.trainKNN()
#knn.classifier('test.png')
'''分别调用并输出原图'''
