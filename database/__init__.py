from database.databasetemplate import DataBaseTemplate
from database.differentdatabase import DataBaseOriginal, SummaryDataBase, FeaturesDataBase, ModelsDataBase


"""
数据库结构介绍
DataBaseTemplate：数据库基类

不同的数据子类
DataBaseOriginal：链接项目原有数据
SummaryDataBase：链接各填充后数据库，共有九个数据库，分别为【训练集、测试集、验证集】和【填充零值、填充中位数、填充类别中中位数】的排列组合
FeaturesDataBase：链接计算后的特征数据库，每个因子一个表，columns为特征，index为时间
ModelsDataBase：链接用于模型的数据库，一个时间点一个表，columns为特征，index为不同因子

ModelDataBase中的数据表，lag_0为时间节点下一个月的数据，属于未来数据，作为因变量使用：["yield_lag_0", ”std_lag_0"]
"""

