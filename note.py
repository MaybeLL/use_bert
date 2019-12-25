# json.dumps()用于将字典形式的数据转化为字符串，json.loads()用于将字符串形式的数据转化为字典
import json

data = {
    'name' : 'Connor',
    'sex' : 'boy',
    'age' : 26
}
data = ['haha','nihao','yes']
print(data)
data1=json.dumps(data)
print(data1)
print(type(data))#输出原始数据格式
print(type(data1))
print(data1[2])

