import pandas as pd

def read_and_clean_data(file_path):
    """ 读取数据并进行数据清洗 """
    df = pd.read_excel(file_path)
    print("原始数据的行列数:", df.shape)

    # 删除前两列
    df.drop(df.columns[[0, 1]], axis=1, inplace=True)

    # 1. 删除第一行
    df = df.iloc[1:]

    # 2. 删除任何含有 "No Data" 的行
    df = df[~df.apply(lambda x: x.astype(str).str.contains('No Data').any(), axis=1)]

    # 将 "磨煤机A 电流" 列的非数值转换为 NaN
    df['磨煤机A 电流'] = pd.to_numeric(df['磨煤机A 电流'], errors='coerce')

    # 3. 删除 "磨煤机A 电流" 列小于 10 的行, NaN 都被视为 False，所以也将 Arc Off-line 删除
    df = df[df['磨煤机A 电流'] >= 10]
    print("清理后数据的行列数:", df.shape)

    return df

def normalize_dataframe(df):
    """ 对 DataFrame 的每一列应用归一化 """
    for column in df.columns:
        # 只对数值类型的列进行归一化
        min_value = df[column].min()
        max_value = df[column].max()
        df[column] = (df[column] - min_value) / (max_value - min_value)
    
    return df



if __name__ == '__main__':
    file_path = "./副本磨煤机取数1分钟.xlsx"
    df_cleaned = read_and_clean_data(file_path)
    df_normalized = normalize_dataframe(df_cleaned)

    # 展示数据处理后的部分数据
    print(df_normalized.head())

    output_file_path = 'data.xlsx'  # 替换为你希望保存的文件名和路径
    df_normalized.to_excel(output_file_path, index=False)

    # 对于树形算法（如决策树、随机森林、梯度提升机）来说，归一化通常不会带来性能提升，因为这些算法不是基于特征的尺度来决定分割点的。





