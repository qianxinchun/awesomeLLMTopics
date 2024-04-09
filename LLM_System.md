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

# 3. 如何评价微软开源的分布式训练框架deepspeed？
https://www.zhihu.com/question/371094177/answer/1058062905
"主要是为数据并行“节省内存“做了一系列很聪明的创新（在这篇论文之前，我们团队的小伙伴儿也发现了类似的技巧，遗憾并没有发表）。节省内存来自两方面（细节可以看论文）：1，对optimizer 阶段的重构，从原来上来就allreduce，各个节点都保持一份weight, gradient以及adam 需要mean, var等，变成只在一个节点保存，通信变成了reduce和broadcast；2，各个层次的参数生命周期不重叠，每个节点仅保存计算时需要的参数。我并不认同论文最后对数据并行和模型并行的讨论。DeepSpeed实质上仍是数据并行， 和 Nvidia Megatron的模型并行相比有优势，原因是Megatron在不应该用模型并行的地方使用了模型并行。举个例子，如果某一个layer的参数量巨大，大到一块GPU装不下，或者即使装的下，使用DeepSpeed 通信量也比模型并行高的话，模型并行仍是最优选择。zero 可以认为理解成数据并行，不过是把参数sharding到多个设备上去，当需要完整参数时，再从其它设备取过来。和tensor model parallelism 不一样"
![image](https://github.com/qianxinchun/awesomeLLMTopics/assets/7309139/aa1c022e-9fe4-4613-b991-475dd8247535)

# 4. 扒一扒Nvidia大规模分布式训练框架Megatron-LM的坑和优化点？
https://www.zhihu.com/question/633778272/answer/3388811917
### 关于LLM结构中Tied Embedding的相关思考
https://zhuanlan.zhihu.com/p/667504988
tied embedding在和zero1结合的时候，不能通过reduce-scatter做最后一步的overlapping gradient reduce，会在最后同步word embedding和lm head梯度的时候造成错位，这种情况下使用all-reduce代替reduce-scatter，性能损失还是相当严重的

### 谈一谈Distributed Optimizer(ZERO)坑爹的地方
https://www.zhihu.com/question/633778272/answer/3388811917
目前LLM大语言模型训练框架中无一例外都使用ZERO1优化器进行训练，能够大幅度降低设备内存占用且不会增加GA（Gradient Accumulation）过程中的开销，因为除了最后一个micro-step，其他step不会产生额外的通信开销，如果是ZERO2-3，那就全程坑爹了，这里就不得不问候一下阿里团队开源的Megatron-LLAMA的核心优化（overlapped zero2），虽然进一步降低了设备内存占用，但是带来的使用限制（GA过程中会有额外的gradient通信，适用于GA较少的场景，对于超大模型就算是千卡集群GA都不会小）显然已经掩盖了它唯一的优点！
有点跑题了……其实综合看下来也就只能ZERO1了，瑕不掩瑜，综合性能还是很扛打的，那么为什么说它坑爹呢？？？
原因有如下几点：
ZERO1会存在DP组梯度的all-reduce/reduce-scatter，由于魔改了DDP，所以一般实现没有跟最后一个step的backward做overlap，然后是sharded parm-state更新完参数后需要all gather所有的parameters，这一步常见的实现也是没有跟下一个batch的第一个step的前向做overlap的。上图！！！
2. 现在部分框架已经能够支持一个batch的最后一个step的backward同all-reduce/reduce-scatter做分桶overlap，但是非常坑爹的是，对于较小模型，如7B/13B，在DP组越来越大的时候，对应bucket的通信是越来越慢的，多次切分通信开销大于单次。那么大到一定程度上，overlap的加速效果就慢慢没有了，到了最后就坑了，反而更慢了。具体数据就不贴了，实在是不好截图，自己复现一下就行哈……
3. 除此之外，还有一个神坑，那就是DP组越大，bucket分的越细，最后在聚合处理时GPU的空闲时间越长，下一个通信算子的时间也会变长，这就是不太能忍了，说啥也不能让GPU空闲不是

### 原始问题的回复

题主提到的两个例子，其实都是调试相关的问题。做系统的人，扒一扒源码，改一改配置，很快就能解决了。比如tied embedding在大模型里基本不会开启，zero stage1的bucket size开得大一点可以提高通信效率。如果dp group size太大导致参数切得太碎，还可以像fsdp一样将shard group和dp group解耦。当然，这两个例子也反映出megatron最大的坑其实在于系统太过臃肿，真正重要的参数藏在一堆无关紧要的特性里，对于不懂系统的人来说还是太难用了。比如在没有配置好pretrain脚本的情况下，算法背景的人想预训练一个标准的llama模型都很麻烦。这可能也解释了为什么开源社区大多选择更加简单易用的fsdp/zero stage3来训练大模型。如果是用到模型并行，deepspeed-megatron要比megatron用得更多，大概也是因为最早gpt-neox是基于前者跑通的，降低了其它用户调试的成本。哪怕是面向系统背景的用户，很多大模型预训练团队也选择了“重新造轮子”，开发适合自己场景的分布式训练系统，例如huggingface的nanotron。毕竟对于transformer大模型而言，实现分布式训练并不难，更重要的是方便自己开发、调试和维护。
