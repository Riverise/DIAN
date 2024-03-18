# DIAN
共包含四个python文件
* utils.py内实现了自定义的accuracy/precision/recall/f1_score等评价指标，以及读取数据集、分批使用数据集的功能函数
* 1.MLP.py内完成第一题MNIST数据集分类任务
* 2.RNN.py内完成第二题FashionMNIST数据集分类任务
* 3.Attention.py内完成了第三题MHA/MQA/GQA的实现

# 学习记录：
## 第一题
运行程序即可得到不同条件下（网络层数、神经元个数、训练步数）的实验结果，实验结果也写在文件末尾的注释中了
此前都是看别人的代码比较多，但轮到自己写，还是能收获不少。
比如data_iter的功能本质上是由python自带的关键字yield实现的。通过yield，可以实现每次迭代时，函数返回特定数量的样本。
