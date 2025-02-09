from torch import nn


class DNN(nn.Module):

    def __init__(self):
        #需要实现父类的__init__方法，不然会报错
        super().__init__()
        #整个全连接神经网络放在Sequential这个容器之中
        self.layer = nn.Sequential(
            #bias:设置是否存在偏移量，默认为有
            #使用Linear实现全连接神经网络的一层
            #图片的原尺寸为28*28，转化为784，输入层为784，输出层为512
            nn.Linear(784,512),
            nn.LayerNorm(512),
            #使用ReLU激活函数进行激活
            nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.Sigmoid(),
            # nn.GELU(),

            #nn.ReLU(),nn.Sigmoid(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,10),
            #最后一层使用Softmax作为激活函数，返回十个概率值
            #NV结构：在进入网络的数据集向上面测试数据集时，是一个批次的进入
            #需要将第二个维度的数据进行激活
            nn.Softmax(dim=1)
        )

    def forward(self,x):

        x=x.reshape(x.shape[0],784) # 64*1*28*28 -> 64*784
        return self.layer(x)

