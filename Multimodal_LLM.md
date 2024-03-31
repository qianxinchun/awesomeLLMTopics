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

# 5. 我们与 GPT-4V 的距离 https://zhuanlan.zhihu.com/p/686257072




# 5. Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning https://github.com/FuxiaoLiu/LRV-Instruction https://arxiv.org/pdf/2306.14565.pdf





