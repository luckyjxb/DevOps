# coding=utf-8
#http://python.jobbole.com/32876/
import itertools
'''创建一个列表（list）时，你可以逐个地读取里面的每一项元素，这个过程称之为迭代（iteration）。'''
mylist = [1, 2, 3]
for i in mylist:
    print i
'''mylist是一个可迭代对象。当使用列表推导式（list comprehension）创建了一个列表时，它就是一个可迭代对象：'''
mylist = [x * x for x in range(30)]
for i in mylist:
    print i
'''生成器（Generators）

生成器也是一个迭代器，但是你只可以迭代他们一次，不能重复迭代，因为它并没有把所有值存储在内存中，而是实时地生成值：'''
mygenerator = (x * x for x in range(3))
for i in mygenerator:
    print i

'''
Yield

Yield是关键字，它类似于return，只是函数会返回一个生成器。
'''
class Bank():# 创建银行，构建ATM机，只要没有危机，就可以不断地每次从中取100
    crisis = False
    def create_atm(self):
        while not self.crisis:
            yield "$100"


hsbc = Bank() # when everything's ok the ATM gives you as much as you want
corner_street_atm = hsbc.create_atm()
print (corner_street_atm.next())

print(corner_street_atm.next())

print([corner_street_atm.next() for cash in range(5)])

# hsbc.crisis = True # 危机来临，没有更多的钱了
print(corner_street_atm.next())
wall_street_atm = hsbc.create_atm() # 即使创建一个新的ATM，银行还是没钱
print(wall_street_atm.next())
hsbc.crisis = False # 危机过后，银行还是空的，因为该函数之前已经不满足while条件
print(corner_street_atm.next())
brand_new_atm = hsbc.create_atm() # 必须构建一个新的atm，恢复取钱业务
# for cash in brand_new_atm:
#     print cash

'''Itertools是你最好的朋友

itertools模块包含一些特殊的函数用来操作可迭代对象。
曾经想复制一个生成器？两个生成器链接？
在内嵌列表中一行代码处理分组？不会创建另外一个列表的Map/Zip函数？
你要做的就是import itertools 。
无例子无真相，我们来看看4匹马赛跑到达终点所有可能的顺序：
'''

horses = [1,2,3,4]
races = itertools.permutations(horses)
print(horses)
print(list(itertools.permutations(horses)))