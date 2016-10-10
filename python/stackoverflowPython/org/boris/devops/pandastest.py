#coding=utf8
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt


#先要安装
#http://www.jb51.net/article/78667.htm
def __printline__(titles='',data=None):
    print '###############################################################'
    print titles
    print data
    print '==============================================================='
s = pd.Series([1,3,5,np.nan,6,8])
print s

dates = pd.date_range('20130101',periods = 6)
print dates
df = pd.DataFrame(data = np.random.randn(6,4),index = dates,columns = list('ABCD'))
print df

df2 = pd.DataFrame({'A':1.,
                    'B':pd.Timestamp('20130102'),
                    'C':pd.Series(1,index = list(range(5)),dtype='float32'),
                    'D':np.array([3]*5,dtype = 'int32'),
                    'E':pd.Categorical(["test","train","test","train","boris"]),
                    'F':'foo'})
print df2

print df2.dtypes
#1、  查看frame中头部和尾部的行：
__printline__(titles = '1、  查看frame中头部和尾部的行：',data = df.head())
__printline__(titles = '查看frame中头部和尾部的行：',data = df.tail(3))
#2、  显示索引、列和底层的numpy数据：
__printline__(titles='显示索引、列和底层的numpy数据：',data=df.index)
#3、  describe()函数对于数据的快速统计汇总：
__printline__(titles = 'describe()函数对于数据的快速统计汇总：',data = df.describe())
#4、  对数据的转置：
__printline__(titles = '对数据的转置：',data=df.T)
#5、  按轴进行排序
__printline__(titles = '按轴进行排序',data = df.sort_index(axis=1,ascending=False))
#6、  按值进行排序
#print df.sort(columns='B',by=False)
__printline__(titles = '按值进行排序',data = df.sort_values(by='C',ascending=False))

#三、选择
print '----------------------------------------------------------'
print '虽然标准的Python/Numpy的选择和设置表达式都能够直接派上用场，但是作为工程使用的代码，我们推荐使用经过优化的pandas数据访问方式： .at, .iat, .loc, .iloc 和 .ix详情请参阅Indexing and Selecing Data 和 MultiIndex / Advanced Indexing。'

print '**************获取****************************'
#获取
#1、 选择一个单独的列，这将会返回一个Series，等同于df.A：
__printline__(titles = '选择一个单独的列，这将会返回一个Series，等同于df.A',data = df['A'])

#2、 通过[]进行选择，这将会对行进行切片

__printline__(titles = '通过[]进行选择，这将会对行进行切片',data = df[0:3])
__printline__(titles = '通过[]进行选择，这将会对行进行切片',data = df['20130102':'20130104'])
print '**************通过标签选择****************************'
# 通过标签选择
#1、 使用标签来获取一个交叉的区域
__printline__(titles='使用标签来获取一个交叉的区域',data = df.loc[dates[0]])

#2、 通过标签来在多个轴上进行选择
__printline__(titles = '通过标签来在多个轴上进行选择',data = df.loc[:,['A','B']])

#3、 标签切片
__printline__(titles='3、 标签切片',data = df.loc['20130102':'20130104',['A','B']])

#4、 对于返回的对象进行维度缩减
__printline__(titles='4、 对于返回的对象进行维度缩减',data=df.loc['20130102',['A','B']])
#5、 获取一个标量
__printline__(titles = '5、 获取一个标量',data=df.loc[dates[0],'A'])
#6、 快速访问一个标量（与上一个方法等价）
__printline__(titles = '6、 快速访问一个标量（与上一个方法等价）',data = df.at[dates[0],'A'])