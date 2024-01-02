## 简单的操作手册

1、创建开发环境，名称、版本自行替换

```js
# 创建环境
conda create --name Langchain-Chatchat python=3.11
# 生效环境
conda activate Langchain-Chatchat

```

2、 安装依赖

```js
# 安装依赖
pip install -r requirements.txt
```

3、 初始化数据

```js
# 初始化本地知识库
python init_database.py --recreate-vs
```

4、启动

```js
# 启动
python startup.py -a --lite
```


5、操作测试

```text
查询用户92V31O6295RT916GJEK91D9FR60A32的保单信息

查询92V31O6295RT916GJEK91D9FR60A32名下保单号为62300504643的保单信息

查询92V31O6295RT916GJEK91D9FR60A32的保单信息

保险的定义

投保人的定义

什么是分红险

East 系统是做什么的
```