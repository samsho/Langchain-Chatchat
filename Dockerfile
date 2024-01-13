# pull the base image
FROM python:3.11.0
LABEL maintainer="SINOSOFT" version="1.0.1"
ARG INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
ARG TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
ARG DEFAULT_TIMEOUT=6000

# order workdir
RUN mkdir -p /app/langchain
COPY ../../../Library/Containers/com.tencent.xinWeChat/Data/Library/Application%20Support/com.tencent.xinWeChat/2.0b4.0.9/d21c596aabacb8ba6005a1755e1e9d22/Message/MessageTemp/fb307cd1951e85f5a1ab59a45bce79bf/File /app/langchain
RUN cd /app/langchain
WORKDIR /app/langchain

# order data
VOLUME ["/app/langchain/knowledge_base","/app/langchain/configs","/app/langchain/logs"]

# upgrade pip 
RUN python -m pip install --upgrade pip

# install dependences
RUN pip install -i ${INDEX_URL} --trusted-host ${TRUSTED_HOST} --default-timeout=${DEFAULT_TIMEOUT} --no-cache-dir --no-deps -r requirements_local.txt

# init database
RUN python init_database.py --recreate-vs

# port 
EXPOSE 8501
 
# start 
CMD ["python", "startup.py","-a","--lite"]