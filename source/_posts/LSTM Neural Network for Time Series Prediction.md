---
title: 利用LSTM进行时间序列预测
date: 2018-01-24 17:54:07
tags: deep learning
categories: 深度学习
---

**本文基于[LSTM Neural Network for Time Series Prediction](http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction)**
***
神经网络是当今机器学习中的新潮流。因此，在基本的神经网络上有过多的课程和教程，从简单的教程到深入描述它们工作的复杂文章。

对于更深层的网络，对图像分类任务的痴迷似乎也使教程出现在更复杂的卷积神经网络上。如果你做的是这样的事情，是极好的，但对我来说，我并不特别热衷于分类图像。我对时间框架的数据更感兴趣。这就是复发神经网络（RNNs）来的方便（我猜想通过阅读这篇文章你会知道长时间的记忆网络（LSTM）是RNN最受欢迎和最有用的变体，如果不是，有很多有用的文章描述了LSTM，可以先去看看）。
<!-- more -->

现在虽然有很多关于LSTM的公共研究论文和文章，但是我发现几乎所有这些都涉及到他们背后的理论运作和数学，他们所提供的例子并不能真正显示超前预测的能力，根据LSTM时间序列。再次，如果您想知道LSTM的错综复杂的工作原理是极好的，但如果您只是希望获得运行正常运行，则不是理想的。

那么我将在这里做的是给出一个关于使用LSTM来预测一些时间序列的完整的代码教程，使用Python[2.7]的Keras包。

友好警告：如果您正在寻找一个从数学和理论的角度来处理LSTM的工作的文章，那么会让你很失望。然而，如果你正在寻找一个有实际编码示例的文章，请继续阅读...

注意：该项目的完整代码可以在[GitHub页面](http://localhost:8888/tree/LoadForecast/LSTM-Neural-Network-for-Time-Series-Prediction-master)上找到。

***
## 一个简单的正弦波
我们可以从我们想到最基本的时间序列开始：正弦函数。让我们创建的数据我们需要的LSTM网络训练在这个功能很多振荡模型。我制作了一个excel电子表格，使幅度和频率为1的sin波（给出6.28的角频率），并且我使用该功能获得超过5001个时间段的数据点，时间差值为0.01。结果就像这样。
![image](http://www.jakob-aungiers.com/img/article/lstm-neural-network-timeseries/sindata.png)
***5001时间段的完整正弦波数据集***

为了节省您使这个的麻烦,我已经把数据放到这个CSV，将使用的训练/测试文件在[这里](https://raw.githubusercontent.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction/master/sinwave.csv)。

现在我们有数据，我们实际试图实现什么？很简单，我们希望LSTM从我们提供的一组窗口数据中学习正弦波，然后希望我们可以要求LSTM来预测这个系列中的下一个N步骤，并且不断地吐出正弦波。

我们将首先将CSV文件中的数据转换并加载到将提供LSTM的numpy数组。Keras LSTM层的工作方式是采用三维数组（N，W，F），其中N是训练序列的数量，W是序列长度，F是每个序列的特征数。我选择了一个序列长度（读取窗口大小）为50，它允许网络在每一个序列中看到正弦波的形状，因此希望自己能根据接收到的窗口建立一个序列的模式。这些序列本身是滑动窗口，因此每次移动1，导致与先前的窗口经常重叠。
![image](http://www.jakob-aungiers.com/img/article/lstm-neural-network-timeseries/sinwindow.png)
***长度为50的序列的示例***

以下是将训练数据CSV加载到适当形状的numpy数组中的代码：

```python
def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'rb').read()
    data = f.decode().split('\n')

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test]
```

接下来我们需要实际构建网络本身。这是简单的部分！至少如果你使用Keras，就像堆积乐高砖一样简单。我使用了[1，50，100，1]的网络结构，其中我们有一个输入层（由一个大小为50的序列组成），其输入具有50个神经元的LSTM层，然后将其馈送到另一个LSTM层，其中100个神经元然后以具有线性激活功能的1个神经元的完全连接的正常层进行馈送，这将用于给出下一个时间步长的预测。

以下是模型构建函数的代码：

```python
def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_shape=(layers[1],layers[0]),
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model

```
最后是时候对数据进行网络训练，看看我们得到了什么。我只用了这个LSTM的1个训练时期，这与传统网络不同，传统的网络需要大量的时代来为网络训练大量的训练示例，在这个1个时期，LSTM将循环遍历训练集中的所有序列窗口。如果这个数据具有较少的结构，则需要大量的时代，但是由于这是具有映射到简单函数的可预测模式的正弦波，训练时期将足够好以获得非常好的近似值全功能功能。

我们将所有这些运行代码放在一个单独的run.py模块中，并运行它：

```python
epochs  = 1
seq_len = 50

print('> Loading data... ')

X_train, y_train, X_test, y_test = lstm.load_data('sp500.csv', seq_len, True)

print('> Data Loaded. Compiling...')

model = lstm.build_model([1, 50, 100, 1])

model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=epochs,
    validation_split=0.05)

predicted = lstm.predict_point_by_point(model, X_test)

```
如果您是观察者，您将在上面的load_data（）函数中注意到，我们将数据分为训练集/测试集，就像机器学习问题的标准做法。然而，我们需要注意的是我们实际想在时间序列的预测中实现的。

如果我们使用测试集，我们将运行每个窗口的真实数据，以预测下一个时间步。如果我们只想预测一个时间步，那么这是很好的，但是如果我们想要预测多个时间步，也许想预测任何紧急的趋势或功能（例如sin函数在这种情况下）使用完整的测试集，将意味着我们会预测下一时间步。无论这一预测，随后的步骤和时间，只使用真实的数据为每个时间步。

您可以在下面的图表中看到，使用这种方法只能预测每个时间点的前一个时间步长：
![image](http://www.jakob-aungiers.com/img/article/lstm-neural-network-timeseries/sinpointprediction.png)
***epochs* = 1, window size = 50**

然而，如果我们想要做出真正的魔法，并预测许多时间步，我们只使用测试数据的第一个窗口作为启动窗口。在每个时间步，我们将窗口后面的最旧的条目弹出，并将下一个时间步长的预测附加到窗口的前面，实质上是将窗口移动，从而缓慢地建立自己的预测，直到窗口充满了预测值（在我们的情况下，由于我们的窗口大小为50，这将在50个时间步长之后发生）。然后，我们无限期地保持这一点，预测下一次对未来时间步长的预测的时间步长，希望看到新趋势。

下图显示正弦波时间序列，仅从真实测试数据的初始起始窗口预测，然后预测约500步：
![image](http://www.jakob-aungiers.com/img/article/lstm-neural-network-timeseries/sinseqprediction.png)
***epochs* = 1, window size = 50**

通过真实的数据，我们可以看到，只有1个时期和一个相当小的数据训练集，LSTM已经做了很好的预测sin的功能。您可以看到，随着我们预测越来越多的未来，误差幅度会随着以前预测中的错误被越来越多地用于将来的预测而增加。因此，我们看到，LSTM没有得到正确的频率，它越来越多地尝试预测它。然而，由于sin函数是具有零噪声的非常容易的振荡函数，它可以很好的预测它。

接下来，我们将尝试看看当我们尝试预测更多随机现实世界的数据时会发生什么。。
*** 

# 一个不那么简单的股票市场
我们预测了一个精确逐点的几百次的正弦波。那么我们现在可以在股市时间序列上做同样的事情呢？

好吧，不行
> **“没有人知道股票会上涨，下跌，横盘还是他妈的圈子” - 马克·汉娜**

不幸的是，股票时间序列不是可以映射的函数。它可以更好地描述为随机游走，这使得整个预测的事情相当困难。但是，LSTM如何识别任何潜在的隐藏趋势呢？那我们来看看吧。

这是一个CSV文件，我在2000年1月至2016年8月期间采取了标准普尔500指数的调整后收盘价。我已经删除了所有内容，使其与我们的正弦波数据完全相同，现在我们将会通过与我们在同一个训练集/测试集的sin波上使用的相同模型运行它。

然而，我们需要对我们的数据进行一些微小的改变，因为正弦波已经是一个很好的标准化重复模式，它通过网络良好地运行原始数据点。然而，通过网络运行股票指数的调整回报将使优化过程自身失效，并且不会收敛到任何类型的最佳数量。所以为了解决这个问题，我们将采取每个n尺寸的训练/测试数据窗口，并对每个窗口进行归一化，以反映该窗口开始时的百分比变化（因此点i = 0处的数据总是为0）。在预测过程结束时，我们将使用以下等式来归一化和随后解规范化，以便将预测中的真实世界数字：


n =价格变动的正常化列表[窗口] 

p =调整后的每日回报价格的原始列表[窗口]

我们在我们的代码中添加了一个**normalise_windows（window_data）** 函数，并更新**了load_data（filename）** 函数，以包括一个条件调用，并取得序列长度并标准化**load_data（filename，seq_len，normalise_window）**：

```python
def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'rb').read()
    data = f.decode().split('\n')

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

```
现在已经如上所述归一化了Windows，因此我们现在可以通过我们的LSTM网络运行我们的库存数据。让我们看看它是如何做的：
![image](http://www.jakob-aungiers.com/img/article/lstm-neural-network-timeseries/stockpointprediction.png)
***epochs = 50, window size = 50***

如上所述在单个逐点预测中运行数据可以很好地匹配回报。但这是欺骗性的！为什么？那么如果你看得更紧密，预测线就是由单一的预测点组成的，这些预测点在他们之前有整个真实历史窗口。因此，网络不需要太多关于时间序列本身的知识，除了每个下一个点最有可能不会离最后一点太远。所以即使它得到了错误的预测，下一个预测将会因真实的历史而忽视不正确的预测，而再次允许出现错误。

我们不能看到LSTM的大脑发生了什么，但是我会强调，对于这个本质上是随机游走的预测（我们已经做了一个完全随机的数据行走）是模仿股票指数的外观，而且完全相同的事情也是如此！）是以基本上高斯分布“预测”下一个点，从而允许基本上随机的预测不会从真实的数据中流失太多。

那么，如果我们想看看在价格走势中是否真的有一些潜在的模式可以辨别，那么我们会怎么看？那么我们会做出与波浪问题相同的事情，让网络预测一系列的点，而不是下一个点。

我们现在可以看到，与作为一个与真实数据几乎相同的正弦波序列的正弦波不同，我们的股票数据预测非常迅速地收敛于某种平衡。 

看看我们所运行的两个训练样本的平衡（一个有1个时期，一个具有50个时期），我们可以看到两者之间存在着巨大的差异。这种疯狂的差异似乎与你所期望的完全一致；通常更高的时代将意味着更准确的模型，然而在这种情况下，它几乎看起来好像单个时代模型倾向于通常遵循短时间价格变动的某种逆转。

我们进一步调查这一点，将我们的预测序列限制到50个未来时间步长，然后每次将启动窗口移动50个，实际上创建了50个时间步长的许多独立序列预测：

我会在这里诚实地说，上面的图表中的结果令我有点惊讶。我期待能够证明，这将是一个愚蠢的游戏，试图从纯粹的历史价格走势来预测未来价格变动的股票指数（因为有这么多潜在因素影响每日价格波动;从基础公司的基本因素，宏观事件，投资者情绪和市场噪音...）然而，检查上面非常有限的测试的预测，我们可以看到，对于很多运动，特别是大型运动，似乎有很大的共识模型预测和随后的价格走势。

我会在这里放一个他妈的大警告标志！为什么上述“有希望的”图可能是错误的，有很多原因。抽样错误，纯粹的运气在一个小的样本大小...这个图表中没有什么应该采取面值，盲目追随一个钱吸吮坑没有一些彻底和广泛的一系列的backtests（这超出了本文的范围）。你已经被警告了

实际上，当我们看看上面的图表同样的运行，但是时代增加到400（这应该使模型模式准确），我们看到，实际上现在只是尝试是预测几乎每个时间段的向上的动力！

然而，我希望你们所有渴望的年轻人都学会了什么使得LSTM网络成为可能的基础，以及如何使用它来预测和映射时间序列，以及这样做的潜在缺陷。

LSTM使用目前在文字预测，AI聊天应用程序，自驾车等众多领域都很丰富。希望本文扩展了在时间序列方法中使用LSTM的实际应用，您已经发现它很有用。

为了完整，下面是您可以在GitHub页面上找到的完整项目代码：

为了参考，我用于运行我的神经网络模型的机器是我强烈推荐的小米Mi笔记本电脑13，因为它具有内置的Nvidia GeForce 940MX显卡，可以与Tensorflow GPU版本一起加速并发模型像LSTM。