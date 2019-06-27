# Industrial_defect_detection
本项目用深度学习的方法进行工业产品缺陷检测，替代原本人眼的产品质检。从而大幅提升工业产品合格率和降低人力成本。

## 1.requirement
- python >= 3.6
- pytorch >= 1.0
- numpy
- PIL


## 2.dataset
数据包括训练数据和测试数据, 共9个类别, 存放在dataset目录中, train.txt为训练数据列表, text.txt为测试数据列表。


## 3.train
### 3.1 修改config.py配置文件
```
配置文件中数据集路径配置为自己的路径
```

### 3.2 执行训练
```
$ python train.py
```

## 4.eval
在测试集上测试模型的acc

只需指定dataroot, testlist, checkpoint参数即可 。例如
```
$ python eval.py /path/to/dataset /path/to/dataset/test.txt checkpoint_resnet_avgpool/checkpoint.path.tar
```

## 5.改进

采用resnet50作为baseline, test acc 为 Test Acc: 0.8687

改进downsample层
原始的残差块downsample层shortcut采用的是stride=2的1x1卷积, 丢失了部分信息, 这里将shortcut修改为2x2, stride=2的AvgPool加stride=1的1x1卷积。=> Test Acc: 0.8751   
