# Deepseek-R1
- Paper: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- Github: [deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)
- Scaled up the RLVR on coding, mathematics, science, logic reasoning (=모두 verifiable rewards가 있는 tasks)
- System Prompt
  <img width="772" height="202" alt="image" src="https://github.com/user-attachments/assets/5e21e783-5b06-4f06-bb55-ee2af45101a7" />


### 중요 Point
- **data를 얻어내는 과정 (생성, 필터링)이 가장 중요한 포인트**
  - Cold Start Long CoT Data, Reasoning Data, Non-Reasoning Data, Combined SFT Data
- R1을 훈련시키는 핵심 data를 가지고 DeepSeek-R1-Distill-Qwen/Llama-*B를 학습시킴 (Distillation)
  - "중간 사고 과정"을 알려주는 data를 큰 DeepSeek 모델이 만듦 (= 어떤 식으로 사고하면 좋은 것이다! 라는 data 생성). 그 data를 작은 model에게 알려줌
- **non-linear reasoning working!** model이 사고하면서, 이전에 진행했었던 step을 다시 검토하고 틀렸으면 수정하고 를 직접 안 알려줌에도 불구하고 자연스럽게 발현이 됨
  - 즉, non-linear / linear reasoning 발현 여부가 중요한 판단
  - 질문: 그럼 왜 이전에는 non-linear reaosning이 안됬을까? (__Why did RLVR suddenly work?__) 답변: [Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs](https://arxiv.org/abs/2503.01307)
    - 4가지 Cognitive behavior를 정의 (Verifications, Subgoal setting, Backtracking, Backward Chaining) 및 측정 → Qwen은 함. Llama은 안함. → 그 후, RLVR를 학습했을 때 위의 행동들을 Qwen만 CoT 길이도 길어지고 성능도 훨씬 올라감. 또한, Llama에게 cognitive behavior를 가르치면 RL이 잘 동작함.
    - 결론: **Llama2와 Llama3은 non-linear reasoning에 필요한 cognitive behavior가 존재하지 않았기 때문이다.**
  - 질문: 왜 그러면 Llama에는 없었을까? 답변: Llama의 train dataset에는 non-linear reasoning이 없었기 때문! (추정: Llama는 OpenWebMath와 FineMath dataset을 사용하지 않았음) [Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs](https://arxiv.org/abs/2503.01307)

### 아쉬운 Point
- "필터링 하는 mechanism" (섬세하게 좋은 CoT data를 골라내는 것)은 공유 안 함

<br>

## 분석 Blog
- [DeepSeek-R1에 관한 비주얼 가이드](https://tulip-phalange-a1e.notion.site/DeepSeek-R1-189c32470be2801c94b6e5648735447d)

<br>

## Code
- huggingface/Fully open reproduction of DeepSeek-R1: https://github.com/huggingface/open-r1
- Building DeepSeek R1 from Scratch: https://github.com/FareedKhan-dev/train-deepseek-r1
