---
title: python语法基础
date: 2025/3/17
tags: 
- 深度学习
- 计算机考研复试
- python
categories: 
- 深度学习
- 李哥考研复试项目
- python
---
# 数据结构
## 整型/浮点型/字符串
```python
a = 3 #整型
a = 3.0 #浮点型
name = "martin"
```
### 使用函数str()避免类型错误
```python
age = 23
message = "Happy" + str(age) + "rd Birthday!"
print(message)
```
这里，调用函数str()，python将非字符串值表示为字符串
## 列表
关键词：list \[]
```python
list1 = [1, 2, 3, 4, 5]
print(list1)
print(list1[2]) # 3
print(list1[-1]) # 5
```
list\[-1]将访问列表最后一个值，-2为倒数第二个值，以此类推。

```python
list2 = [1, "art", dict1] #同一个列表中支持多种数据类型
```
## 字典（哈希表）
关键词： dict {key: value}
```python
dict1 = {"name": "martin", "age": 18, 20: 80}
print(dict1["name"])
print(dict1["age"])
print(dict1[20])
```

# 运算
`//` 地板除
`/` 普通除
`**` 乘方
```python
print(9/2) # 4.5
print(9//2) # 4
print(9**2) # 81
```

# 函数
带初值的函数示例
```python
def func1(a, b = 2) #如果传入b，返回a**b；否则，返回a**2
	return a ** b

print(func1(2)) # 4
```

# 列表
## 切片
```python
list1 = [1, 2, 3, 4, 5]
print(list1[1:4]) # [2, 3, 4] 左闭右开
print(list1[1:-1]) # [2, 3, 4]
print(list1[:]) # [1, 2, 3, 4, 5]
print(list1[1:]) # [2, 3, 4, 5]
```
## 操作
### 添加元素
```python
list1.append(6) # [1, 2, 3, 4, 5, 6] 在末尾添加
list1.insert(0, 0.5) #[0.5, 1, 2, 3, 4, 5] 在指定位置添加
list1.extend([8, 9]) #在列表末尾一次性追加另一个列表的多个值，即用新列表扩展原来的列表
```
### 删除元素
```python
del list1[0] #删除索引0处的元素，且不再以任何方式使用它
list1.pop(index) #删除索引index处的元素，但会继续使用它，index为空时默认为-1，即栈顶
list1.remove(key) #删除列表中值为key的元素，也可继续使用它的值；若列表中有多个值为key，remove只删除第一个指定的值
```
### 列表排序
```python
list1.sort() # 永久顺序排序
list1.sort(reverse = true) # 永久倒序排序
list1.sorted() # 临时顺序排序
```
## 数值列表
range()
```python
for value in range(1,5):
	print(value)
```
将打印数字1 2 3 4. 左闭右开
要创建数字列表，可使用函数list()将range()的结果直接转换为列表
```python
list(range(1,5))
print(list) # [1, 2, 3, 4]
```
函数range()还可指定步长
```python
even_numbers = list(range(2,11,2))
print(even_numbers) # [2, 4, 6, 8, 10]
```

# 字典
## 遍历字典
```python
favorite_languages = {
	'jen':'python',
	'sarah':'c',
	'edward':'ruby',
	'phil':'python',
}
# 遍历所有的键值对
for key, value in favorite_languages.items()
# 遍历字典中所有的键
for key in favorite_languages.keys()
# 遍历字典中所有的值
for value in favorite_languages.values()
# 按顺序遍历
for key in sorted(favorite_languages.keys())
# 去重遍历
for value in set(favorite_languages.values())
```
# 类
## 类编码风格
类名应采用驼峰命名法，即将类名中的每个单词的首字母都大写，而不使用下划线。实例名和模块名都采用小写格式，并在单词间加上下划线。
对于每个类，都应紧跟在类定义后面包含一个文档字符串。这种文档字符串简要地描述类的功能，并遵循编写函数的文档字符串时采用的格式约定。每个模块也都应包含一个文档字符串，对其中的类可用于做什么进行描述。
可使用空行来组织代码，但不要滥用。**在类中，可使用一个空行来分隔方法；而在模块中，可使用两个空行来分隔类。**
需要同时导入标准库中的模块和你编写的模块时，先编写导入标准库模块的import语句，再添加一个空行，然后编写导入你自己编写模块的import语句。在包含多条import语句的程序中，这种做法让人更容易明白程序使用的各个模块都来自何方。
## 创建和使用类
### 创建类
```python
class Dog():
	def __init__(self, name, age):
		"""初始化属性name和age"""
		self.name = name
		self.age = age

	def sit(self):
		"""模拟小狗被命令时蹲下"""
		print(self.name.title() + " is now sitting.")

	def roll_over(self):
		"""模拟小狗被命令时打滚"""
		print(self.name.title() + " rolled over!")
```
- 根据约定，在python中，首字母大写的名称指的是类。
- 方法__init__()
	- 类中的函数称为方法。
	- 方法__init__()是一个特殊的方法，每当你根据Dog类创建新实例时，python都会自动运行它。在这个方法的名称中，开头和末尾各有两个下划线，这是一种约定，旨在避免python默认方法与普通方法发生名称冲突。
	- 在这个方法的定义中，形参self必不可少，还必须位于其他形参的前面。因为python在调用这个__init__()方法来创建实例时，将自动传入形参self。每个与类相关联的方法调用都自动传递实参self，它是一个指向实例本身的引用，让实例能够访问类中的属性和方法。本例，我们创建Dog实例时，python将调用Dog类的方法__init__()。我们将通过实参向Dog()传递名字和年龄；self会自动传递，因此我们不需要传递它。每当我们根据Dog类创建实例时，都只需给最后两个形参(name和age)提供值。
	- 定义的两个变量都有前缀self。以self为前缀的变量都可供类中的所有方法使用，我们还可以通过类的任何实例来访问这些变量。self.name = name获取存储在形参name中的值，并将其存储到变量name中，然后该变量被关联到当前创建的实例。self.age = age的作用与此类似。像这样可通过实例访问的变量称为属性。
- Dog类还定义了另外两个方法：sit()和roll_over()。由于这些方法不需要额外的信息，如名字和年龄，因此它们只有一个形参self。
### 根据类创建实例
```python
# 创建实例 my_dog
my_dog = Dog('Heymi', 4)
# 访问属性
my_dog.name
# 调用方法
my_dog.sit()
```
## 使用类和实例
### 给属性指定默认值
类中的每个属性都必须有初始值，哪怕这个值是0或空字符串。在有些情况下，如设置默认值时，在方法__init__()内指定这种初始值是可行的；如果你对某个属性这样做了，就无需包含为它提供初始值的形参。
```python
class Car():
	def __init__(self, make, model, year):
		"""初始化描述汽车的属性"""
		self.make = make
		self.model = model
		self.year = year
		self.odometer_reading = 0 #python将创建一个名为odometer_reading的属性，并将其初始值设置为0	
```
### 修改属性的值
```python
my_new_car = Car("audi", "a5", 2025)
# 直接修改属性的值
my_new_car.odometer_reading = 23
# 通过方法修改属性的值
class Car():
	--snip--
	
	def update_odometer(self, mileage):
		"""将里程表读数设置为指定的值"""
		self.odometer_reading = milege
my_new_car.update_odometer(23)
# 通过方法对属性的值进行递增
class Car():
	--snip--
	
	def increment_odometer(self, miles):
		"""将里程表读数增加指定的量"""
		self.odometer_reading += miles
```
## 继承
一个类继承另一个类时，它将自动获得另一个类的所有属性和方法；原有的类称为父类，而新类称为子类。子类继承了其父类的所有属性和方法，同时还可以定义自己的属性和方法。
### 子类的方法__init__()
```python
class ElectricCar(Car):
	"""电动汽车的独特之处"""

	def __init__(self, make, model, year):
		"""初始化父类的属性"""
		super().__init__(make, model, year)
```
### python 2.7中的继承
```python
class Car(object):
	--snip--

class ElectricCar(Car):
	def __init__(self, make, model, year):
		super(ElectricCar, self).__init__(make, model, year) # 1
```
1 函数super()需要两个实参：子类名和对象self。
在python 2.7中使用继承时，务必在定义父类时在括号内指定object

### 给子类定义属性和方法
```python
class Car():
	--snip--

class ElectricCar(Car):
	def __init__(self, make, model, year):
		super().__init__(make, model, year)
		self.battery_size = 70 # 1
```
1 添加了新属性self.battery_size，并设置其初始值。根据ElectricCar类创建的所有实例都将包含这个属性，但所有Car实例都不包含它。
### 重写父类的方法
对于父类的方法，只要它不符合子类的行为，都可对其进行重写。为此，可在子类中定义一个与要重写的父类方法同名的方法。这样，python将不会考虑这个父类方法，而只关注你在子类中定义的方法。
### 将实例用作属性
使用代码模拟实物时，你可能会发现自己给类添加的细节越来越多：属性和方法清单以及文件都越来越长。在这种情况下，可能需要将类的一部分作为一个独立的类提取出来。
例如，不断给ElectricCar类添加细节时，可能其中包含很多专门针对Battery的属性和方法，则可将这些属性和方法提取出来，放到一个名为Battery的类中，并将一个Battery实例用作ElectricCar类的一个属性：
```python
class Car():
	--snip--

class Battery():
	def __init__(self, battery_size=70): # 1
		self.battery_size = battery_size

class ElectricCar(Car):
	def __init__(self, make, model, year):
		super().__init__(make, model, year)
		self.battery = Battery() # 2
```
1: \_\_init__()除self外，还有另一个形参battery_size。这个形参是可选的：如果没有给它提供值，电瓶容量将被设置为70。
2: 在ElectricCar类中，我们添加了一个名为self.battery的属性。这行代码让python创建一个新的Battery实例（由于没有指定尺寸，因此为默认值70），并将该实例存储在属性self.battery中。每当方法__init__()被调用时，都将执行该操作；因此现在每个ElectricCar实例都包含一个自动创建的Battery实例。
# 导入
## 导入函数
```python
import module_name # 导入整个模块
module_name.function_name() # 使用模块中的函数需要使用句点

from module_name import function_name # 导入特定函数，该函数后续使用时不需要句点

import pizza as p # 使用as给模块指定别名
```

## 导入类
```python
from car import Car, ElectricCar # 在模块文件car.py中导入Car类、ElectricCar类
import car # 导入整个car模块
my_beetle = car.Car('volkswagen', 'beetle', 2025) # 创建类实例代码都必须包含模块名，即需要使用句点访问
```

# 注释
Python 中的注释有**单行注释**和**多行注释**。
## 单行注释
单行注释以 # 开头，例如：
```python
#这是一个注释
print(hello, world)
```
## 多行注释
多行注释用三个单引号 ''' 或者三个双引号 """ 将注释括起来，例如
### 单引号
```python
#!/usr/bin/python3 
'''
这是多行注释，用三个单引号
这是多行注释，用三个单引号 
这是多行注释，用三个单引号
'''
print("Hello, World!")
```
### 双引号
```python
#!/usr/bin/python3 
"""
这是多行注释（字符串），用三个双引号
这是多行注释（字符串），用三个双引号 
这是多行注释（字符串），用三个双引号
"""
print("Hello, World!")
```
## 拓展说明
在 Python 中，多行注释是由三个单引号 ''' 或三个双引号 """ 来定义的，而且这种注释方式并不能嵌套使用。
当你开始一个多行注释块时，Python 会一直将后续的行都当作注释，直到遇到另一组三个单引号或三个双引号。
**嵌套多行注释会导致语法错误。**
例如，下面的示例是不合法的：
```python
'''
这是外部的多行注释
可以包含一些描述性的内容

    '''
    这是尝试嵌套的多行注释
    会导致语法错误
    '''
'''
```
在这个例子中，内部的三个单引号并没有被正确识别为多行注释的结束，而是被解释为普通的字符串。
这将导致代码结构不正确，最终可能导致语法错误。
如果你需要在注释中包含嵌套结构，推荐使用单行注释（以#开头）而不是多行注释。
单行注释可以嵌套在多行注释中，而且不会引起语法错误。例如：
```python
'''
这是外部的多行注释
可以包含一些描述性的内容

# 这是内部的单行注释
# 可以嵌套在多行注释中
'''
```
这样的结构是合法的，并且通常能够满足文档化和注释的需求。