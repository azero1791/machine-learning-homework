#!/usr/bin/env python 
# 指定bash中使用python interpreter

import matplotlib.pyplot as plt # TODO: 作图
import numpy as np # TODO: 方便计算 
from matplotlib.font_manager import FontProperties # TODO: 设置中文字体
font = FontProperties(fname="SimHei.ttf", size=14)

# TODO: 闭式解
def closure_solve(x, y):
    args = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)   # TODO: 使用np库提供的线性代数计算获得闭式解的参数向量

    def predict(predict_x):
        y = np.dot(np.array([1, predict_x]), args)  # TODO: 使用学习后的模型预测给定年份的房价
        return y
    return predict # TODO: 使用function curring返回函数

# TODO: 梯度下降学习获取数学模型
def gd_solve(tmp_x, y):
    x = tmp_x
    x = (tmp_x - np.mean(tmp_x))/np.std(tmp_x)
    learning_rate=0.001 # 学习率
    epochs=10000 # 迭代轮数
    m = len(x)
    a, b = 0.0, 0.0  # 初始化参数

    for _ in range(epochs):
        predictions = a + b * x
        d_a = (1 / m) * np.sum(predictions - y) # TODO:计算梯度的a分量
        d_b = (1 / m) * np.sum((predictions - y) * x) # TODO:计算梯度的b分量
        a -= learning_rate * d_a # TODO:更新a
        b -= learning_rate * d_b # TODO:更新b

    def predict(p_x):
        a_x  = (p_x - np.mean(tmp_x)) / np.std(tmp_x) 
        p_y = a + b * a_x
        return p_y
    return predict #TODO: 函数curring返回预测模型
def main():
   x = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013] # 时间
   v_x = [[1, e]for e in x] # TODO: 构造x的向量
   y = [2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365, 5.704, 6.853, 7.971, 8.561, 10.000, 11.280, 12.900] # 房价

   plt.scatter(x, y, c = 'blue', label = '训练集')  # TODO: 绘制2000年~2014年房价的散点图(训练集) 
    

   # 假设线性模型:y=a*1+b*x
   # 闭式解
   # TODO: 获取闭式解的数学模型并预测2014年的房价
   closure_predict = closure_solve(np.array(v_x), np.array(y))
   predict_y = closure_predict(2014) 
   print(f"闭式解的预测:2014年的房价为{predict_y:.3f}")

   # TODO: 绘制闭式解学习后的数学模型
   plt.plot(x, [closure_predict(e) for e in x], c='red', label='闭式解的预测模型') 


   # TODO: 梯度下降解
   gd_predict = gd_solve(np.array(x, dtype=np.float64), np.array(y, dtype=np.float64))
   print("梯度下降预测2014年房价: {:.3f}".format(gd_predict(2014)))
   # TODO: 绘制梯度下降解学习后数学模型
   plt.plot(x, [gd_predict(e) for e in x], c='green', label='梯度下降的预测模型')

   # TODO: 添加标签, 标题, 图例
   plt.xlabel('年份', fontproperties=font)
   plt.ylabel('房价', fontproperties=font)
   plt.title('南京市平均房价训练集及预测模型', fontproperties=font)
   plt.legend(prop=font)

   # TODO: 显示图形
   plt.show()
   return

if __name__ == "__main__":
    main()
