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
pip install --upgrade --no-deps -r requirements-sm.txt
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



## 镜像打包

确保环境已安装docker并且镜像源改为国内的（如：https://docker.mirrors.tuna.tsinghua.edu.cn）。

进入Dockerfile所在目录执行如下命令构建进行与启动。

```dockerfile
# 构建
docker build -t langchain:1.0.1 .
# 查看
docker images
# 运行 ip地址为服务器的内网地址
docker run -d -p 172.17.0.13:8501:8501 --name langchain langchain:1.0.1
# 推送到远程仓库（可选，换成自己的仓库地址）
docker login registry.cn-hangzhou.aliyuncs.com
docker tag langchain:1.0.1 registry.cn-hangzhou.aliyuncs.com/motry/langchain:1.0.1
docker push registry.cn-hangzhou.aliyuncs.com/motry/langchain:1.0.1
```

