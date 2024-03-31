# 1. Transformer升级之路：17、多模态编码位置的简单思考 https://kexue.fm/archives/10040
![image](https://github.com/qianxinchun/awesomeLLMTopics/assets/7309139/fbfe50a9-054a-47d8-af67-408e2958247a)

# 2. “闭门造车”之多模态模型方案浅谈 https://kexue.fm/archives/9984
![image](https://github.com/qianxinchun/awesomeLLMTopics/assets/7309139/ae424930-37fd-4199-a937-76dfe8379167)
![image](https://github.com/qianxinchun/awesomeLLMTopics/assets/7309139/b6294df9-66e0-41f5-8fb8-8726d3f3008a)
![image](https://github.com/qianxinchun/awesomeLLMTopics/assets/7309139/c9875780-52bb-4428-9f89-0fd43866281d)

# 3. 在用llava架构训vlm时，llm基模选择base模型好还是chat模型好呢？ https://www.zhihu.com/question/650838721/answer/3448817012
原问题：看很多模型都是用base，但像mobilevlm用的chat模型效果指标也挺好。而且llava1.6的34b基模也是在Yi34b上finetune过的。所以有人做过实验测试vlm用base或chat的差别吗？
“对多模态大模型的LLM基座来说，base模型更好，经过SFT的模型更容易产生带幻觉的冗长回答。这个观察来自于Stanford的Percy Liang团队最近的论文Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models，此文探索了多模态大模型设计的诸多关键要素，题主的问题——LLM基座选base还是chat是其中重要的一点。可以看出，使用经过SFT的Vicuna-v1.5模型并不能带来显著的性能提升，而且从以下的定性cases中可以看出使用Vicuna的模型倾向于产生更冗长、幻觉程度更严重的答案，因此作者推荐在训练视觉-语言大模型时使用base版本的LLM作为文本侧的基座模型。”
“个人觉得benchmark只能做为参考的一部分，一个好的vlm给用户的体感很重要，要得到generalization且instruction-following比较强的vlm还是得在chat模型上。benchmark的问题（instruction）形式都比较简单，很容易理解，跟user问的各种模糊且奇葩问题完全是两个概念。”

# 4. Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models https://arxiv.org/pdf/2402.07865.pdf https://github.com/TRI-ML/prismatic-vlms https://github.com/TRI-ML/vlm-evaluation
With these requirements,
we implement our training codebase in PyTorch, using Fully
Sharded Data Parallel (FSDP; Zhao et al., 2023) and BF16
mixed precision. FSDP lets us specify precision for individual model components (e.g., FP16 for vision backbones,
BF16 for LMs), enables portability to different hardware,
and provides minimal implementation overhead. We
observe 20% faster step times with our FSDP-backed implementation, a notable gain given LLaVa leverages the welloptimized DeepSpeed ZeRO library. 

## the same choices used by many other recent VLMs: “letterbox padding” to process images
因此，不改变原图宽高比的前提下对图像进行缩放是很有必要的，这就是letterbox padding的用处。
执行letterbox padding时通常遵循以下步骤：
计算目标分辨率与原始图像之间的宽高比差异。
根据宽高比，确定哪个维度（宽度或高度）需要完全匹配目标尺寸，而另一个维度则按比例缩放，以避免图像内容扭曲。
将图像缩放到这样的尺寸，使宽度或高度完全匹配目标分辨率，而另一维度的长度小于或等于目标分辨率。
在较短的一维两侧添加条纹（"paddings"），这些条纹通常是黑色，但也可以是其他颜色或图案，以此来填充缺失的空间，使得处理后的图像符合目标分辨率。
例如，如果要将16:9的视频帧调整为1:1的正方形形状，就可以在视频帧的上下或左右添加等宽的条纹。

![image](https://github.com/qianxinchun/awesomeLLMTopics/assets/7309139/2bea6c1f-0277-4365-82cc-eee69c4db34f)

![image](https://github.com/qianxinchun/awesomeLLMTopics/assets/7309139/6092738f-cdb4-42e6-a933-87122c8694b6)

![image](https://github.com/qianxinchun/awesomeLLMTopics/assets/7309139/319f5795-2227-4f2b-8bf1-5d583ddc0c3f)
## Experiments – Investigating Design Axes
（1） We find (Fig. 4; left) that including the explicit projector pretraining stage is unnecessary,
with single-stage training improving aggregate performance
(p = 0.007). Eliminating this first stage saves 20-25% of
training cost, and removes the need for additional, stagespecific data (e.g., the captioning subset from §2). As this
change strictly improves performance and efficiency, we
adopt single-stage training for all following experiments.

（2）we ask – is there potential to improve VLM performance by
finetuning the full model, including the visual backbone?
We find (Fig. 5) that this is not the case, and that finetuning the visual backbone significantly degrades performance. The degraded performance from full finetuning
could be for a number of reasons ranging from the scale
and diversity of the vision-language data we train on to
language generation as a learning objective (vs. objectives
that encourage learning fine-grained perceptual features).

（3）We find
that the backbones trained with vision-language contrastive
objectives (i.e., CLIP and SigLIP) are significantly more
performant than alternatives (p = 8.88e-8). while cropping is clearly suboptimal, the
“naive resize” scheme is the most performant for CLIP. For
SigLIP, both “naive resize” and “letterbox padding” perform
similarly. In general, our results favor “naive resizing” over
“letterbox padding” but we cannot rule the improvement
statistically significant (p = 0.0148). An image with a 16:9 aspect ratio that
is padded to square introduces a large amount of uninformative pixels (exceeding 40%); warping the aspect ratio is
possibly less of a shift. Coupled with the innate patch dimensionality of a Vision Transformer (d = 1024 for a 16 × 16
pixel patch), naively resizing an image may preserve enough
information for the downstream LM (with 7B+ parameters)
to extract the properties necessary for downstream tasks.
![image](https://github.com/qianxinchun/awesomeLLMTopics/assets/7309139/a04b2107-4ed6-4932-b19c-cf0933973cbc)

（4）A rich
body of prior work in vision identifies that different types
of visual representations trained with different inductive
biases can lead to improved performance for a broad spectrum of applications (Kobayashi et al., 2022; Karamcheti
et al., 2023).  We
find (Fig. 7 - left) that fusing DINOv2 and SigLIP features
provides significant gains across the board (p = 0.00162),
with a notable exception for the DINOv2 + CLIP models
(p = 0.4066), where combining DINOv2 features seem to
be particularly harmful on TextVQA. Looking at the remaining results, we see especially impressive gains of 5-10% on
localization and challenge tasks; in general, the DINOv2 +
SigLIP fused representations are the most performant visual
representations we try, with virtually no added parameters. 

（5） Unfortunately,
instruct tuning has drawbacks, introducing bias and regressions in performance (Ouyang et al., 2022). We find (Fig. 7 - right) that instruction-tuned LMs yield no
statistically significant improvement in performance over
base LMs (p = 0.373), but differ in qualitative performance.
Specifically, we observe that instruct-tuned LMs lead to
VLMs that are more verbose, prone to hallucination, and
generally less specific in their responses (Fig. 11).  However, given that the language-only data is the only
source of “safety” data during finetuning, we explicitly
probe our VLMs with directly offensive and toxic prompts,
to evaluate how important this data is for inducing safeguards on VLM outputs. In our adversarial testing, we find
that especially for VLMs trained from base LMs such as
Llama-2, including this co-training data is important for
inducing at least a minimal set of safeguards.

（6）Unsurprisingly, we find (Fig. 10; middle) evidence
of severe underfitting with a single epoch, with steady improvement (especially for tasks requiring structured outputs
such as RefCOCO) until two epochs, when performance
plateaus. We find that training for two epochs yields a significant increase in improvement over training for one epoch. We find (Fig. 10; right) that adding both datasets
improves performance (p = 0.0138), but that LRV-Instruct
offers most of the resulting performance gain, an indication
that sourcing diverse images will be increasingly important
to scaling future VLMs.

（7） Of
primary concern is the generality of our model architecture;
while the three component architecture we define in §2 is
reflective of the majority of existing VLMs, there are other
architecture innovations and optimization procedures that
our study does not currently capture; as a notable example, we do not study the Perceiver-based architectures used
by models such as Flamingo or IDEFICS (Alayrac et al.,
2022; Laurenc¸on et al., 2023) for interleaved image-text
training.

（8） Visually-conditioned language models inherit all of the risks
and biases associated with language models (Touvron et al.,
2023; Brown et al., 2020), as well as with underlying vision
models and corresponding pretraining datasets

# 5. 我们与 GPT-4V 的距离 https://zhuanlan.zhihu.com/p/686257072 https://reka.ai/reka-flash-an-efficient-and-capable-multimodal-language-model/
"此外，应对上下文的限制，QFormer、Perceiever 也已经被广泛地验证了其有效性。"

"应用场景广泛：这个也很直接，日常生活中大多数数据的呈现方式就是，图片 + 文本 -> 文本的范式能够极大扩充模型处理任务的范围。另外，随着大语言模型发展催生出的一系列 Agent 研究，在浏览网页的时候会依赖 html 作为输入。如果能够直接让 Agent 看到屏幕，输出对应的操作坐标，更加简洁优雅。进一步地，Deepmind 的 RT 2 也验证了视觉语言模型能够很快地迁移到诸如 robotic 场景，在 embodied 环境中发挥重要的作用。"

"我们也同样可以用这一条路径来进一步验压缩即智能这一想法，看看这一框架是否能够在具备了更丰富模态信息后，背后世界模型的学习速率是否会进一步加快。"

"模态桥接（Modality Bridge）：负责将视觉编码器得到的图片表示进行变换映射到新的空间方便 LLM 进行处理。这里的做法有一些不同的方案：
(1) Image as Word Embedding：一种经典的尝试是将视觉编码器得到的图片向量通过简单的 MLP 映射到对应的 word embedding 维度，随后就可以将图片作为多个 word embeddings 进行统一的处理。这一方面的局限是视觉编码器的分辨率往往是固定且比较小的(224px 和 336px)。而在很多场景下这样的分辨率完全不够用（OCR 识别、Agent浏览等），可以通过 post-training 来提升图片的分辨率也可以 bypass 掉 image encoder（没有了预训练分辨率的限制），直接将图片切成小块，随后映射到 word embedding 空间，Fuyu-8B 就是这样一个思路，在高分辨率的场景下展现出了非常出色的性能。分辨率提升带来的图片向量数量平方级增长带来的计算开销，可以通过利用 QFormer 或者是 Perceiver 来映射到固定数量来解决。
(2) Cross Attention to Visual Embedding: Deepmind 最早搞的 Flamingo 就是通过在 LLM 中引入额外的 Gated Cross-Attention Layer，来在文本生成的过程中整合视觉端的信息. 这种方案对区分不同模态有着更加强的先验，但后续看到的一些开源实现和改进，都很难超越前一种方案。如果训练量足够大，那么在前一种方案中 LLM 也能够自适应地学习到这种先验，因而我个人觉得这个方案或许在 2 年前是有道理，但在今天 scaling law 的暴力美学下，可能更少先验，更多数据会是朴实且有效的方案。
（3）还有一种是 adaptive 的搜索式的方案 V*，根据 query 需要来切分出图片里的小块重新交给模型，类似起到 re-attention 的效果，在小物体的检测问题上面有很大的潜力。总的来说，这些方案都是为了解决输入图片分辨率不够的问题，一个大/自适应分辨率的视觉编码器可能是更本质的解法。"
![image](https://github.com/qianxinchun/awesomeLLMTopics/assets/7309139/b544c4a0-157c-4ac0-b154-24f380329465)

![image](https://github.com/qianxinchun/awesomeLLMTopics/assets/7309139/2412227c-6d77-4607-8ca2-78278809c7ee)

## 训练数据
### Alignment Dataset
一种解决思路是对 alignment数据集进行更加细粒度的表述，进而能够帮助模型更好地学习图片中物体的相关位置等关系，和LLM原先的知识挂上钩。ShareGPT4V 就是一个很好的尝试，验证了利用 GPT-4V 重新标注 image captions，就能够带来明显的提升。除了 ShareGPT4V 以外，CapsFusion 也展现了用更丰富的 caption （https://huggingface.co/datasets/BAAI/CapsFusion-120M CAPSFUSION: Rethinking Image-Text Data at Scale https://github.com/baaivision/CapsFusion  However, our experiments reveal significant Scalability Deficiency and World Knowledge Loss issues in models trained with synthetic captions,
which have been largely obscured by their initial benchmark success. Upon closer examination, we identify the
root cause as the overly-simplified language structure and
lack of knowledge details in existing synthetic captions.
To provide higher-quality and more scalable multimodal
pretraining data, we propose CAPSFUSION, an advanced
framework that leverages large language models to consolidate and refine information from both web-based image-text
pairs and synthetic captions.）带来的提升，并且开源了 100M 的数据集。

from CAPSFUSION
![image](https://github.com/qianxinchun/awesomeLLMTopics/assets/7309139/728753e2-b9e6-49b3-8f3c-b1ab64913f5c)



### SFT Dataset
学术界开源的比较好的训练数据目前主要是 LLaVA 系列，其利用 bounding box 等辅助信息将图片文本化后，利用 ChatGPT/GPT-4 来生成了大量的 pseudo multimodal pair (detailed captioning, reasoning and conversation)。这个范式非常有效，也是为什么 LLaVA 系列一出来效果很惊艳的原因。但他依旧存在着一些问题：

因为 ChatGPT 并没有真正地看到图片内容，其 pseudo multimodal 的数据必然会引发 hallucination 问题。这一点也是目前大家关注的重点。解决的方案有 LLaVA-RLHF ，通过额外引入一个 Factual reward model 来提升 hallucination； Volcano 则是用 self-feedback 来 revise 输出。或者更直接一点，我们可以用早先人工标注的数据做一下统一格式，在保真度方面就会有很大的提升。这方面我们做了 M3IT （https://huggingface.co/datasets/MMInstruction/M3IT https://m3-it.github.io/ We introduce the Multi-Modal, Multilingual Instruction Tuning (M3IT) dataset, comprises 40 carefully curated datasets, including 2.4 million instances and 400 manually written task instructions, reformatted into a vision-to-text structure. Key tasks are translated into 80 languages with an advanced translation system. We train a VLM model on our M³IT dataset, showcasing its potential to answer complex questions, generalize to unseen video tasks, and comprehend unseen instructions in Chinese.），整合了之前很多常用的数据集，方便大家做 SFT 。
任务的覆盖面不够广，在重要的 OCR、Chart 场景下能力都有所欠缺。这点我们对比 Qwen、LLaVA 1.5 以及 LLaVA-Next 的性能就能看出来，使用了更多更丰富的多模态数据集，基本上都能对下游如 MMMU、MathVista 等测评数据集有所提升。


### GPT-4V 背后一定是大量的数据工程
Alignment 端：相比于开源模型利用 CLIP 等作为 vision encoder，OpenAI 应该采用了强化版的 CLIP 模型（毕竟现在的 CLIP 还都是他们 2021 年的成果）。之前的 CLIP 不够好的很大原因就在于图片和文本的信息量不对等，caption 大多是简单的几个词来描述物体，而图片中则有丰富的颜色、位置等时空信息。不妨可以想象一下，我们用现在的 GPT-4V 标注整个 web images（~ 10B level ?），提升文本端的丰富度并对 hallucination 做控制。在此数据集基础上我们训练一个 vision encoder，再迭代地更新 GPT-4V，相信会有一个明显的提升；
SFT 端：我认为在足够好的对齐 + 在基模型足够强大这两个条件下，可能只需要足够多样的（质量 >> 数量）的 prompting 数据就能够在现在的 VQA、Captioning benchmark 上表现出色。因为客观来说，现在的测评数据集也都集中在这两个任务形式上，因此少量的 prompt 就能够泛化到下游的数据集上。


### 特定领域的 Benchmark
hallucination 是多模态更容易体现出来的一个问题，造成的潜在后果也挺大，这方面测评的benchmark 像 POPE 和 MMHal。但是 POPE 有个问题这个数据集依赖于 COCO 的 tag，就我个人的经验而言，那个 tag 的准确率并不高，POPE 上的分数因而会收到一定程度的影响。此外，大家认为 math reasoning 可能是比较有挑战性的任务，因此像 MMMU 和 MathVista 的关注度都比较高，目前 GPT-4V 也距离人类还是有很大差距。这块我们最近的一个工作是意识到 ArXiv 上的很多 paper 天然也是多模态的，并且涵盖了丰富的学科内容，因而我们构建了一个 Multimodal ArXiv，提供 captioning 和 QA (GPT-4V generated）的数据集，能够很有效地提升模型数学推理的能力。






# 6. Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning https://github.com/FuxiaoLiu/LRV-Instruction https://arxiv.org/pdf/2306.14565.pdf





