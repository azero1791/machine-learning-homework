# 数据
x = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013]
y = [2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365, 5.704, 6.853, 7.971, 8.561, 10.000, 11.280, 12.900]

# 数据预处理
import numpy as np

x = np.array(x, dtype=np.float64)
y = np.array(y, dtype=np.float64)

# 将 x 数据标准化（减去均值再除以标准差）
x_mean = np.mean(x)
x_std = np.std(x)
x_normalized = (x - x_mean) / x_std

# 线性回归闭式解
def linear_regression_closed_form(x, y):
    X = np.vstack((np.ones_like(x), x)).T
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta

# 梯度下降
def linear_regression_gradient_descent(x, y, learning_rate=0.001, epochs=10000):
    m = len(x)
    theta_0, theta_1 = 0.0, 0.0  # 初始化参数
    
    for _ in range(epochs):
        predictions = theta_0 + theta_1 * x
        d_theta_0 = (1 / m) * np.sum(predictions - y)
        d_theta_1 = (1 / m) * np.sum((predictions - y) * x)
        theta_0 -= learning_rate * d_theta_0
        theta_1 -= learning_rate * d_theta_1
        
    return theta_0, theta_1

# 闭式解模型
theta_closed_form = linear_regression_closed_form(x_normalized, y)

# 梯度下降模型
theta_gradient_descent = linear_regression_gradient_descent(x_normalized, y, learning_rate=0.001, epochs=10000)

# 预测 2014 年房价（先对 2014 年数据标准化）
year_to_predict = 2014
x_pred_normalized = (year_to_predict - x_mean) / x_std
price_closed_form = theta_closed_form[0] + theta_closed_form[1] * x_pred_normalized
price_gradient_descent = theta_gradient_descent[0] + theta_gradient_descent[1] * x_pred_normalized

# 输出结果
print("闭式解参数: 截距 = {:.4f}, 斜率 = {:.4f}".format(theta_closed_form[0], theta_closed_form[1]))
print("闭式解预测2014年房价: {:.4f}".format(price_closed_form))
print("梯度下降参数: 截距 = {:.4f}, 斜率 = {:.4f}".format(theta_gradient_descent[0], theta_gradient_descent[1]))
print("梯度下降预测2014年房价: {:.4f}".format(price_gradient_descent))
