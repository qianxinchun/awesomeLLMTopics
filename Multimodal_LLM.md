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




