import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib

# 灰度关联分析
def grey_relation_analysis(file_path):
    df = pd.read_excel(file_path)

    X = df.iloc[:, 1:]  # 输入特征：第二列到最后一列
    y = df.iloc[:, 0]   # 目标变量：第一列
    # 标准化数据
    std_X = (X - X.min()) / (X.max() - X.min())
    std_y = (y - y.min()) / (y.max() - y.min())

    # 初始化关联系数矩阵
    grey_matrix = np.zeros(std_X.shape)

    # 计算关联系数
    for i in range(std_X.shape[1]):
        grey_matrix[:, i] = np.abs(std_X.iloc[:, i] - std_y)

    # 计算关联度
    rho = 0.5  # 分辨系数，通常取值在0到1之间
    grey_relation = (np.min(grey_matrix) + rho * np.max(grey_matrix)) / (grey_matrix + rho * np.max(grey_matrix))
    grey_relation_degree = np.mean(grey_relation, axis=0)


    # 将特征关联度与特征名称对应
    feature_names = X.columns
    relation_results = pd.DataFrame({'Feature': feature_names, 'RelationDegree': grey_relation_degree})

    # 打印特征关联度
    print(relation_results.sort_values(by='RelationDegree', ascending=False))

    return grey_relation_degree


def plot_grey_relation_analysis():
      
    # 数据  
    data = {  
        '关键参数': ['磨煤机A 电流.1', '磨煤机A加载油压力', '磨煤机A电机线圈温度3', '给煤机A给煤量',   
                '磨煤机A电机轴承温度1', '磨煤机A一次风流量', 'A磨煤机比例溢流阀开度',   
                '磨煤机A电机轴承温度2', '磨煤机A辊轴承润滑油温度3', '环境温度'],  
        '关联度': [0.870111, 0.863444, 0.833234, 0.796272, 0.790431, 0.755219, 0.743810,   
                0.731221, 0.621829, 0.595534]  
    }  
    matplotlib.rcParams['font.family'] = 'SimHei'  # 指定 'SimHei' 字体，该字体支持中文显示
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    # 创建DataFrame  
    df = pd.DataFrame(data)  
    
    # 绘制条形图  
    bars = plt.bar(df['关键参数'], df['关联度'])  

    # 在每个条形上显示数值  
    for bar in bars:  
        yval = bar.get_height()  
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 3), va='bottom')  # round to 3 decimal places 
    
    # 设置标题和轴标签  
    plt.title('关键参数关联度条形图')  
    plt.xlabel('关键参数')  
    plt.ylabel('关联度')  
    
    # 显示图形  
    plt.show()



def plot_grey_relation_analysis_black():
      
    # 数据  
    data = {  
        '关键参数': ['电流.1', '加载油压力', '电机线圈温度3', '给煤量',   
                '电机轴承温度1', '一次风流量', '比例溢流阀开度',   
                '电机轴承温度2', '辊轴承润滑油温度3', '环境温度'],  
        '关联度': [0.870111, 0.863444, 0.833234, 0.796272, 0.790431, 0.755219, 0.743810,   
                0.731221, 0.621829, 0.595534]  
    }  
    matplotlib.rcParams['font.family'] = 'SimHei'  # 指定 'SimHei' 字体，该字体支持中文显示
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    # 创建DataFrame  
    df = pd.DataFrame(data)  
    
    # 设置图片大小
    plt.figure(figsize=(12, 4))  # 12英寸宽，6英寸高
    
    # 绘制条形图  
    bars = plt.bar(df['关键参数'], df['关联度'])  

    # 在每个条形上显示数值  
    for bar in bars:  
        yval = bar.get_height()  
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 3), va='bottom')  # round to 3 decimal places 
    
    # 设置标题和轴标签  
    plt.title('关键参数关联度条形图')  
    plt.xlabel('关键参数')  
    plt.ylabel('关联度')  
    
    # 显示图形  
    plt.show()

if __name__ == '__main__':
    # grey_relation_analysis(file_path="./feature_importance_model.xlsx")
    # grey_relation_analysis(file_path="./permutation_feature_importance.xlsx")
    # grey_relation_analysis(file_path="./recursive_feature_elimination.xlsx")
    # plot_grey_relation_analysis()
    plot_grey_relation_analysis_black()

