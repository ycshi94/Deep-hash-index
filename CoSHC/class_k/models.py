#coding=utf-8
import torch
import torch.nn.functional as F
class code_Net(torch.nn.Module):     # 继承 torch 的 Module
    def __init__(self, n_feature, n_output):
        super(code_Net, self).__init__()     # 继承 __init__ 功能
        self.hidden1 = torch.nn.Linear(n_feature, 128)   # 隐藏层线性输出
        self.hidden2 = torch.nn.Linear(128, 84)
        self.out = torch.nn.Linear(84, n_output)     # 输出层线性输出

    def getclass(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x1 = F.relu(self.hidden1(x))      # 激励函数(隐藏层的线性值)
        x2=F.relu(self.hidden2(x1))
        x = self.out(x2)                 # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return x

    def forward(self,vec,label):
        out = self.getclass(vec)
        criteria = torch.nn.CrossEntropyLoss()
        loss = criteria(out, label.long())
        return loss
class all_Net(torch.nn.Module):     # 继承 torch 的 Module
    def __init__(self, n_feature, n_output):
        super(all_Net, self).__init__()     # 继承 __init__ 功能
        self.code_hidden1 = torch.nn.Linear(n_feature, 128)   # 隐藏层线性输出
        self.code_hidden2 = torch.nn.Linear(128, 84)
        self.code_out = torch.nn.Linear(84, n_output)     # 输出层线性输出
        self.desc_hidden1 = torch.nn.Linear(n_feature, 128)  # 隐藏层线性输出
        self.desc_hidden2 = torch.nn.Linear(128, 84)
        self.desc_out = torch.nn.Linear(84, n_output)  # 输出层线性输出

    def getclass(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x1 = F.relu(self.code_hidden1(x))      # 激励函数(隐藏层的线性值)
        x2=F.relu(self.code_hidden2(x1))
        x = self.code_out(x2)                 # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return x

    def forward(self,codevec,codelabel,descvec,desclabel):
        codeout = self.getclass(codevec)
        codecriteria = torch.nn.CrossEntropyLoss()
        codeloss = codecriteria(codeout, codelabel.long())
        descout = self.getclass(descvec)
        desccriteria = torch.nn.CrossEntropyLoss()
        descloss = desccriteria(descout, desclabel.long())
        loss=(codeloss+descloss)/2
        return loss