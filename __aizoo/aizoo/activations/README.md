https://www.kaggle.com/siavrez/deepfm-model


https://www.cnblogs.com/shiyublog/p/11121839.html


https://www.cnblogs.com/tangjunjun/p/12839457.html

https://zhuanlan.zhihu.com/p/87696023


https://github.com/digantamisra98/Mish

https://zhuanlan.zhihu.com/p/84418420

https://blog.csdn.net/u011984148/article/details/101444274


注意：优化Mish的学习率很可能会获得更好的结果。本文提出了相对于ReLU而言，较低的学习率供参考。

Mish检查了理想的激活函数应该是什么(平滑、处理负号等)的所有内容，并提出了一套泛化的初始测试。在过去的一年里，我测试了大量新的激活函数，其中大多数都失败了，从基于MNIST的论文，到基于更真实的数据集的测试。因此，Mish可能最终会为深度学习实践者提供一种新的激活功能，并有很大的机会超过长期占据主导地位的ReLU。




Mish的性质
我通过下面的PyTorch代码链接提供了Mish，以及一个修改过的XResNet (MXResNet)，这样你就可以快速地将Mish放入你的代码中，并立即进行测试！

让我们后退一步，了解什么是Mish，为什么它可能改进ReLU上的训练，以及在神经网络中使用Mish的一些基本步骤。

什么是Mish？
直接看Mish的代码会更简单一点，简单总结一下，Mish=x * tanh(ln(1+e^x))。

其他的激活函数，ReLU是x = max(0,x)，Swish是x * sigmoid(x)。