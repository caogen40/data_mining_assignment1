import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体（以SimHei为例）
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

paths_1G = [f"../1G_data/part-{str(index).zfill(5)}.parquet" for index in range(8)]
paths_10G = [f"../10G_data/part-{str(index).zfill(5)}.parquet" for index in range(8)]
paths_30G = [f"../30G_data/part-{str(index).zfill(5)}.parquet" for index in range(16)]

# 删除重复行
df_list = []
for path in paths_30G:
    temp_df = pd.read_parquet(path)
    temp_df = temp_df.drop_duplicates()
    df_list.append(temp_df)
df = pd.concat(df_list, axis=0, ignore_index=True)
df = df.drop_duplicates()

# 检查数据的基本信息
print(df.info())

# 检查缺失值情况
missing_data = df.isnull().sum()
print("\n缺失值统计：")
print(missing_data)

# 删除包含缺失值的列
df = df.dropna()

# 检查处理后的数据信息
print(df.info())
print(df.head(3))

# 处理时间戳
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['registration_date'] = pd.to_datetime(df['registration_date'])

# 解析purchase_history列中的JSON字符串
df['purchase_history'] = df['purchase_history'].apply(lambda x: json.loads(x))

# 绘制年龄分布的KDE曲线
plt.figure(figsize=(10, 6))
sns.kdeplot(df['age'], fill=True)
plt.title('年龄分布的核密度估计曲线')
plt.xlabel('年龄')
plt.ylabel('密度')
plt.tight_layout()  # 紧凑型布局
plt.savefig('age_kde_plot.png', dpi=600)
plt.close()

# 绘制收入分布的KDE曲线
plt.figure(figsize=(10, 6))
sns.kdeplot(df['income'], fill=True)
plt.title('收入分布的核密度估计曲线')
plt.xlabel('收入')
plt.ylabel('密度')
plt.tight_layout()  # 紧凑型布局
plt.savefig('income_kde_plot.png', dpi=600)
plt.close()

# 绘制性别分布
gender_counts = df['gender'].value_counts()
plt.figure(figsize=(10, 6))
gender_counts.plot(kind='pie', autopct='%1.1f%%', title='性别分布')
plt.ylabel('')
plt.tight_layout()
plt.savefig('gender_distribution.png', dpi=600)
plt.close()

# 解析购买历史，提取类别和价格信息
df['purchase_category'] = df['purchase_history'].apply(lambda x: x['category'] if isinstance(x, dict) else None)
df['average_purchase_price'] = df['purchase_history'].apply(lambda x: x['average_price'] if isinstance(x, dict) else None)

# 绘制购买类别分布
purchase_category_counts = df['purchase_category'].value_counts()
plt.figure(figsize=(10, 6))
purchase_category_counts.plot(kind='bar', title='购买商品类别分布')
plt.xlabel('类别')
plt.ylabel('样本量')
plt.tight_layout()
plt.savefig('purchase_category_distribution.png', dpi=600)
plt.close()

# 绘制平均购买金额分布的KDE曲线
plt.figure(figsize=(10, 6))
sns.kdeplot(df['average_purchase_price'], fill=True)
plt.title('平均购买金额分布的核密度估计曲线')
plt.xlabel('平均购买金额')
plt.ylabel('密度')
plt.tight_layout()
plt.savefig('average_purchase_price_kde_plot.png', dpi=600)
plt.close()