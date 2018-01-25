---
title: ubuntu16 环境下 TensorFlow serving安装指南 
date: 2018-01-24 17:54:07
tags: 安装调试
categories: 深度学习
---

![Markdown_photo/blog/20180125003.jpg](https://github.com/fyingzhu/Markdown_photo/blob/master/blog/20180125003.jpg?raw=true)
## 1. 安装bazel
> 安装参考[Install Bazel on Ubuntu](https://docs.bazel.build/versions/master/install-ubuntu.html)

采用第一种 **Using Bazel custom APT repository (recommended)** 方法：
1. Install JDK 8

```
sudo apt-get install openjdk-8-jdk
```
2. Add Bazel distribution URI as a package source (one time setup)

```
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
```
<!-- more -->
3. Install and update Bazel

```
sudo apt-get update && sudo apt-get install bazel
```
Once installed, you can upgrade to a newer version of Bazel with:

```
sudo apt-get upgrade bazel
```

## 2. 安装gprc


```
sudo pip install grpcio
```
提示没有pip，安装pip

## 3.Packages

To install TensorFlow Serving dependencies, execute the following:

```
sudo apt-get update && sudo apt-get install -y \
        build-essential \
        curl \
        libcurl3-dev \
        git \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        python-numpy \
        python-pip \
        software-properties-common \
        swig \
        zip \
        zlib1g-dev
```

## 3. 从源代码安装克隆TensorFlow Serving存储库

```
git clone --recurse-submodules https://github.com/tensorflow/serving
cd serving
```
> clone会非常慢，建议百度加速，我是硬等了3个小时才clone完。 

下一步，配置TensorFlow, 运行：
```
cd tensorflow
./configure
cd ..
```
> 配置过程中，会有很多选项问你是否需要，根据需要输入y/n即可。不用的尽量填n，否则后面会有很多文件的路径难以配置（LZ遇到的坑：有些文件根本没有，却让你输入路径）。

> 配置的第一项是确认python路径，默认的是/usr/bin/python，可以用
which python查看当前python的路径。

## 4. Build
To build the entire tree, execute:

```
bazel build -c opt tensorflow_serving/...
```
To test your installation, execute:
```
bazel test -c opt tensorflow_serving/...
```

***
### 注：过程中会遇到的问题及解决方法：
国外软件源安装遇到的坑会少很多。。所以请首先更换软件源在安装其他软件吧。

- 添加软件源
安装完Ubuntu 16.04后 ，更换为国内的软件源：

```
sudo gedit /etc/apt/sources.list
```


在文件开头添加下面的网易的软件源：

```
deb http://mirrors.163.com/ubuntu/ precise-updates main restricted
deb-src http://mirrors.163.com/ubuntu/ precise-updates main restricted
deb http://mirrors.163.com/ubuntu/ precise universe
deb-src http://mirrors.163.com/ubuntu/ precise universe
deb http://mirrors.163.com/ubuntu/ precise-updates universe
deb-src http://mirrors.163.com/ubuntu/ precise-updates universe
deb http://mirrors.163.com/ubuntu/ precise multiverse
deb-src http://mirrors.163.com/ubuntu/ precise multiverse
deb http://mirrors.163.com/ubuntu/ precise-updates multiverse
deb-src http://mirrors.163.com/ubuntu/ precise-updates multiverse
deb http://mirrors.163.com/ubuntu/ precise-backports main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ precise-backports main restricted universe multiverse
```
更新软件源：

```
sudo apt-get update
```

- 更改文件夹权限

sudo chmod 777 XXX

- 结束当前命令

ctrl + c

## 参考文献
1. [Ubuntu 16.04LTS安装后需要做的事](http://blog.csdn.net/liuqz2009/article/details/52087019)
1. [https://github.com/tensorflow/serving](https://github.com/tensorflow/serving)
2. [https://github.com/grpc/grpc/tree/master/src/python/grpcio](https://github.com/grpc/grpc/tree/master/src/python/grpcio)
3. [Install Bazel on Ubuntu](https://docs.bazel.build/versions/master/install-ubuntu.html)