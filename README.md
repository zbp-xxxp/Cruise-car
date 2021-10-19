# 基于模仿学习的智能车系统

以EdgeBoard嵌入式边缘计算平台为核心处理单元，基于模仿学习使小车根据摄像头采集的图像数据完成自主车道线巡航

# 一、效果展示

## 车道线巡航效果展示（第一人称视角）

b站视频链接：[https://www.bilibili.com/video/BV1vM4y1N7jK/](https://www.bilibili.com/video/BV1vM4y1N7jK/)


## 车道线巡航效果展示（第三人称视角）

b站视频链接：[https://www.bilibili.com/video/BV1rh411z7te/](https://www.bilibili.com/video/BV1rh411z7te/)

## 2021年信创展

我们在2021年信创人才培养论坛活动展会上展示我们的智能车系统

<img style="display: block; margin: 0 auto;" src="https://ai-studio-static-online.cdn.bcebos.com/fa2340201c2444f1a4377c4fef036209129c6607f7d546d18e77dbb6e4cff008" width = "100%" height = "100%" />

# 二、决策数据采集

数据采集的整体架构如下图所示：

<img style="display: block; margin: 0 auto;" src="https://ai-studio-static-online.cdn.bcebos.com/7f53610d4bbb40a4b05100d7f578a30afba5b3941388499f993084a2650bfeb1" width = "50%" height = "50%" />


智能小车上使用标准的广角摄像头作为视觉传感器，其基本参数如下表所示：

<img style="display: block; margin: 0 auto;" src="https://ai-studio-static-online.cdn.bcebos.com/bdfd851dc8424f35a4b46257219db3c999c7a7d595a04ca2bfff15465207bbfc" width = "50%" height = "50%" />

采集的数据已上传至AI Studio：[【模仿学习】车道线巡航决策数据集](https://aistudio.baidu.com/aistudio/datasetdetail/102649/0)


# 三、模仿学习

本项目受端到端机器学习方法的启发，基于卷积神经网络实现智能车系统的车道线巡航，仅使用一个单目摄像头采集车道线数据，通过行为克隆的方法实现车道线自主巡航。

## 1.卷积神经网络

基础的卷积神经网络（Convolutional Neural Network, CNN)[6]由卷积层(convolution)，激活层(activation)和池化层(pooling)三种结构组成。CNN输出的结果是每幅图像的特定特征空间。整个过程最重要的工作就是如何通过训练数据迭代调整网络权重，也就是反向传播算法。目前主流的卷积神经网络，比如VGG，ResNet都是由简单的CNN调整、组合而来。

为了研究卷积神经网络行为克隆方法的可行性，本项目通过卷积神经网络来实现智能车转角信息的预测：

<img style="display: block; margin: 0 auto;" src="https://ai-studio-static-online.cdn.bcebos.com/424a39d8af84450f8f204860d680708bc00cd28b649b4fcb8b6da39ddb8c0902" width = "80%" height = "80%" />

我们建立了一个含有5层卷积层的卷积神经网络模型：

<img style="display: block; margin: 0 auto;" src="https://ai-studio-static-online.cdn.bcebos.com/233c5807428a4c4caa363768c7cf55c54cfa73c7fae3410e97ffd72b2dd018df" width = "80%" height = "80%" />


```python
import paddle as paddle
import paddle.fluid as fluid

# 定义模型
def cnn_model(image):
    temp = fluid.layers.conv2d(input=image, num_filters=32, filter_size=5, stride=2, act='relu')
    temp = fluid.layers.conv2d(input=temp, num_filters=32, filter_size=5, stride=2, act='relu')
    temp = fluid.layers.conv2d(input=temp, num_filters=64, filter_size=5, stride=2, act='relu')
    temp = fluid.layers.conv2d(input=temp, num_filters=64, filter_size=3, stride=2, act='relu')
    temp = fluid.layers.conv2d(input=temp, num_filters=128, filter_size=3, stride=1, act='relu')
    temp = fluid.layers.dropout(temp, dropout_prob=0.1)
    fc1 = fluid.layers.fc(input=temp, size=128, act="leaky_relu")
    fc2 = fluid.layers.fc(input=fc1, size=32, act="leaky_relu")
    drop_fc2 = fluid.layers.dropout(fc2, dropout_prob=0.1)
    predict = fluid.layers.fc(input=drop_fc2, size=1, act=None)
    predict = fluid.layers.tanh(predict / 4)
    return predict
```
## 2.模型训练

训练代码已整理在AI Studio上：[【AI达人创造营EdgeBoard嵌入式部署】基于飞桨实现模仿学习的智能车系统](https://aistudio.baidu.com/aistudio/projectdetail/2489871)

## 3.模型部署

基于PaddleLite完成部署。构建一个模型预测的基类：

```
class Predictor:
    """ base class for Predictor interface"""
    def load(self, j):
        """ load model """
        pass

    def set_input(self, data, index):
        """ set input at given index data is numpy array"""
        pass

    def get_output(self, index):
        """ output Tensor at given index can be cast into numpy array"""
        pass

    def run(self):
        """ do inference """
        pass
```


```python
class PaddleLitePredictor():
    """ PaddlePaddle interface wrapper """
    def __init__(self):
        self.predictor = None

    def load(self, model_dir):
        from paddlelite import Place
        from paddlelite import CxxConfig
        from paddlelite import CreatePaddlePredictor
        from paddlelite import TargetType
        from paddlelite import PrecisionType
        from paddlelite import DataLayoutType
        valid_places = (
            Place(TargetType.kFPGA, PrecisionType.kFP16, DataLayoutType.kNHWC),
            Place(TargetType.kHost, PrecisionType.kFloat),
            Place(TargetType.kARM, PrecisionType.kFloat),
        )
        config = CxxConfig()
        if os.path.exists(model_dir + "/params"):
            config.set_model_file(model_dir + "/model")
            config.set_param_file(model_dir + "/params")
        else:
            config.set_model_dir(model_dir)
        config.set_valid_places(valid_places)
        self.predictor = CreatePaddlePredictor(config)

    def set_input(self, data, index):
        input = self.predictor.get_input(index)
        input.resize(data.shape)
        input.set_data(data)

    def run(self):
        self.predictor.run()

    def get_output(self, index):
        return self.predictor.get_output(index)
```

# 四、自主巡航

将模型部署到EB板后，就可以调用模型，完成自主巡航任务。

## 图像预处理

部署端采集到的图像不能直接送入模型进行预测，在预测前任然需要做数据预处理。


```python
import cv2
import numpy as np

# 图像预处理
def process_image(frame, size, ms):
    frame = cv2.resize(frame, (size, size))
    img = frame.astype(np.float32)
    img = img - ms[0]
    img = img * ms[1]
    img = np.expand_dims(img, axis=0)
    return img
```

## 巡线模型


```python
cnn_args = {
    "shape": [1, 3, 128, 128],
    "ms": [125.5, 0.00392157]
}
cruise_model = "/home/root/workspace/autostart/src/models/cruise" # 模型路径

# CNN网络预处理
def cnn_preprocess(args, img, buf):
    shape = args["shape"]
    img = process_image(img, shape[2], args["ms"])
    hwc_shape = list(shape)
    hwc_shape[3], hwc_shape[1] = hwc_shape[1], hwc_shape[3]
    data = buf
    img = img.reshape(hwc_shape)
    # print("hwc_shape:{}".format(hwc_shape))
    data[0:, 0:hwc_shape[1], 0:hwc_shape[2], 0:hwc_shape[3]] = img
    data = data.reshape(shape)
    return data

# CNN网络预测
def infer_cnn(predictor, buf, image):
    data = cnn_preprocess(cnn_args, image, buf)
    predictor.set_input(data, 0)
    predictor.run()
    out = predictor.get_output(0)
    return np.array(out)[0][0]

class Cruiser:
    def __init__(self):
        hwc_shape = list(cnn_args["shape"])
        hwc_shape[3], hwc_shape[1] = hwc_shape[1], hwc_shape[3]
        self.buf = np.zeros(hwc_shape).astype('float32')
        self.predictor = predictor_wrapper.PaddleLitePredictor()
        self.predictor.load(cruise_model)

    def cruise(self, frame):
        res = infer_cnn(self.predictor,self.buf, frame)
        return res
```

在调用模型时，只需要实例化上面定义的`Cruiser`即可，如下所示：
```
cruiser = Cruiser()
image = carmer.read() # 获取摄像头的输出
angle = cruiser.cruise(front_image) # 获取当前状态下的动作（直走or转弯）
cart.steer(angle) # 根据上一步的结果使小车做出相应的动作
```

# 总结与升华

我们基于模仿学习里的行为克隆方法使智能车系统自主完成车道线巡航任务，通过大量实验以及比赛现场的发挥，我们在第十六届全国大学生智能车竞赛区赛中排第18名（一共81支参赛队伍），并且在全国总决赛中获二等奖。

更多关于本项目的相关报道：
- [请党放心，强国有我！——机器人学院代表学校参加世界机器人大赛再传佳音](https://mp.weixin.qq.com/s/AO1bU8cJhsqZYe4B2LC3ng)
- [喜报：机器人学院第一学生党支部党员多人获得科技竞赛大奖](https://mp.weixin.qq.com/s/YjKNswLuX34pvrgmyEmKkQ)
