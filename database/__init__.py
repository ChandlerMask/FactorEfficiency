from database.databasetemplate import DataBaseTemplate
from database.differentdatabase import DataBaseOriginal, FeaturesDataBase, ModelsDataBase


"""
数据库结构介绍
DataBaseTemplate：数据总表数据库
    1. 链接原始数据：三个csv文件和json文件
    2. 链接填充后数据总表：train、val、test分别一个数据库，每个数据库中保存不同填充方式的数据总表（数据总表与原始csv格式一致）
    
DataBase子类：模型数据库
    1. 每个子类对应一种填充方式，下面链接两个数据库：因子特征数据库与模型数据数据库
    2. 因子特征数据库：每个因子一个数据表
    3. 模型数据数据库：每个时间节点一个数据表
"""

