# 0. LLM（六）：GPT 的张量并行化（tensor parallelism）方案 https://zhuanlan.zhihu.com/p/603908668
图像示例非常清楚
![image](https://github.com/qianxinchun/awesomeLLMTopics/assets/7309139/ed20e4ea-da79-4691-b2cb-258568be6fac)
![image](https://github.com/qianxinchun/awesomeLLMTopics/assets/7309139/972a4a97-4300-470d-8ce2-3945f9f24811)
![image](https://github.com/qianxinchun/awesomeLLMTopics/assets/7309139/ec1ffa1b-0982-4468-939e-3694f9c9505f)
![image](https://github.com/qianxinchun/awesomeLLMTopics/assets/7309139/f900f37b-4db5-4f47-988f-656bd2007cf4)
![image](https://github.com/qianxinchun/awesomeLLMTopics/assets/7309139/8fff9a23-80b5-4065-80a9-10b766df6b11)

# 1. Cross Entropy Loss 的并行化方案 https://zhuanlan.zhihu.com/p/497672789
![image](https://github.com/qianxinchun/awesomeLLMTopics/assets/7309139/05247f8d-5abe-4582-bf89-0b5a8ded8438)


# 2. 模型并行训练：为什么要用Megatron，DeepSpeed不够用吗？ https://zhuanlan.zhihu.com/p/670958880
![image](https://github.com/qianxinchun/awesomeLLMTopics/assets/7309139/fbc9a1f7-0ffa-4138-9eff-0f2ea40e4bab)
容易得出，在BLOOM看来，tensor并行、Fused CUDA Kernels 和 DataLoader 是Megatron相对于 DeepSpeed 的三大特点。
Fused CUDA Kernels
简单来说，就是nvidia对cuda运算的优化，这部分代码在Megatron代码里都是c/c++。
![image](https://github.com/qianxinchun/awesomeLLMTopics/assets/7309139/ce0a13ec-5dbb-47d8-8140-7c60522efa6e)

如图的例子，本来这个计算涉及到三个函数，需要换进换出显存三次。融合后，只需要一次换进换出了。加快了计算速度。

"核心是卡堆到几百张以后必须上megatron里的tp和pp那一套了，deepspeed推的zero123本质还是dp，卡数上去了以后all reduce的通信开销越来越heavy。既不像tp那样能高效利用节点内带宽，也不像pp那样低通信开销然后用各种trick去overlap掉bubble。当然这部分目前我也还没看到有定量计算就是了...
节点内通信快用tp，节点间通信慢用pp，这个比较容易直观理解。那为啥几百张卡还要用dp呢？只用tp、pp不就行了吗？
tp开销比dp大，用tp是为了在一台服务器内存一套完整的参数
我的理解是tp和pp是为了解决模型参数量太大，无法放在单张卡，甚至是一台机器上的多张卡这个问题。当模型参数都已经放在GPU上了，就需要通过DP这种并行方式来加速训练了
### 请教下为什么说zero123本质上还是dp？
zero12是将optimizer、梯度分到不同卡上。zero3是将模型参数的存储分到不同卡上，实际计算的时候，每张卡跑不同的数据，跑到模型某一层的时候，把这一层的模型参数从其他卡传过来，进行计算。所以实际计算的时候还是数据并行，但是zero3在用不到模型某些参数的时候，就把他们放到其他卡上存着。
"
![image](https://github.com/qianxinchun/awesomeLLMTopics/assets/7309139/f4eed334-38b7-47c6-b29e-a0da3d836767)

3.Megatron中tensor并行数、pipeline并行数、数据并行数等各并行参数如何确定取值范围？
主要参考：Efficient Training on Multiple GPUs

这里面讨论的情况比较复杂，我的个人理解和总结如下：

基于给定模型、给定输入长度、给定机器的情况下。

Tensor并行数
tensor并行无气泡等额外计算；但是同步的数据量大，需要快的网络，因此都是在单个机器上；模型太大，单个层在gpu放不下，也得tensor并行。

放得下单层 <= tensor并行数 <= 单机卡数

Pipeline并行数
pp通信快，适合通信慢模型大到单卡/单机放不下的场景；存在气泡；tensor并行放不下，再用pipeline并行。

多大机器能放的下模型参数，这个得具体算一下，计算方法可参考：回旋托马斯x：分析transformer模型的参数量、计算量、中间激活、KV cache

通过micro batch来减少pipeline并行的气泡，提高训练效率。

放得下模型参数 <= tensor并行数 * pipeline并行数 。 pipeline并行数通常能整除层数，或者考虑embedding 层的时候能整出(层数+2)。

数据并行数
有更多机器，就数据并行；数据并行常用deepspeed zero数据并行，stage需要额外注意。

数据并行数 = 机器数 / tensor并行数 / pipeline并行数。



补充：文档中主要是推荐了一些取值范围，像是一个节点上有8个机器的时候，到底tensor并行数为2、4还是8，这个可能还是得看具体实验的结果；这个问题，目前我也没有理的很清楚，感觉还是得后面做一些具体实验，才能对这个问题有更深入、清晰的理解。

4.架构分层
从主流大模型对于并行策略的使用可以看出，基本都使用了两三种并行策略，为啥没人只用一种策略呢？

个人觉得这个跟GPU机器网络结构本来就是分层的有关。

基本都是多机N卡，这些模型训练用的卡数基本都是8。单机8卡之间的通信较快，而不同机器之间的通信速度较慢。

如果采用一种并行训练策略，例如数据并行，那么一次梯度同步，需要所有卡进行通信，其中同一个机器上的通信已经完成，而不同机器上卡的通信还要一段时间，这浪费了机器内的带宽。

因此对应于分层架构，主要是分层次的通信速度/带宽，需要采用分层的并行策略。最常见的就是机器内采用tensor 并行，机器外采用其他并行方案。


5.Megatron v2论文中和zero3的比较
论文名称： Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM

PTD-P是pipeline、tensor、data parallelism三种方式的组合。

从论文的比较来看，同样的模型、同样的机器资源下，PTD-P 可以获得比 只用zero3更好的训练效率，当2240卡时，能差出三倍多来。

For example, by doubling the number of GPUs (keep- ing the batch size the same), PTD-P outperforms ZeRO-3 by 70% for both models due to less cross-node communication
论文中觉得主要原因是zero3的跨节点通信更多，就是我们“架构分层“讨论的差不多。PTD在这里的优势就是tensor并行走的都是节点内的通信，数据并行、pipeline并行走节点间通信。而zero3每次都要节点间通信，自然就慢了。

当然这个是megatron的实验结果，也不能全信，换个人同样问题的实验就可能给出不一样的结论。例如一个问题，是他这里设置了固定的batch size，那多卡的实际操作的时候为了更好的结果，显存充足，利用率低完全可以增加micro batch size，那可能结论就不一样了。（= =！我也没有资源，我也只能保持怀疑态度，不能偏听偏信）

we cannot use data parallelism in isolation for very large models with a limited training batch size because of a) insuffi- cient memory capacity, and b) scaling limitations of data parallelism (e.g., GPT-3 was trained to convergence with a batch size of 1536. Data parallelism thus supports parallelization to only 1536 GPUs; however, roughly 10, 000 GPUs were used to train this model in a reasonable amount of time).
v2这篇论文也提到了不能只用数据并行的两个原因：

1.数据并行占用显存大。当然这个是说普通的数据并行，zero123显然降低了显存的使用

2.只数据并行时的 batch size >= 卡数，且是卡数的倍数。假设万卡，我就想batch size=100，只数据并行就不行了。
