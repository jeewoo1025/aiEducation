# Role-Prompting: Does Adding Personas to Your Prompts Really Make a Difference?

본 글은 [Role-Prompting: Does Adding Personas to Your Prompts Really Make a Difference?](https://www.prompthub.us/blog/role-prompting-does-adding-personas-to-your-prompts-really-make-a-difference) 을 참조하여 4가지 논문에 대한 정리글입니다. 


## What is Role-Prompting (a.k.a Persona Prompting)?
LLM에게 특정 역할(ex. scientist)을 부여하여 그 역할에 맞는 어휘, 지식 수준, 시각으로 응답하도록 유도하는 기법이다. 이를 통해 모델의 응답이 보다 일관되고 특정 도메인에 맞는 정보 제공이 가능하다. 

## Research

### When "A Helpful Assistant" Is Not Really Helpful: Personas in System Prompts Do Not Improve Performances of Large Language Models
- link: https://arxiv.org/pdf/2311.10054
- 저자: Mingqian Zheng, Jiaxin Pei, Lajanugen Logeswaran, Moontae Lee, David Jurgens
- 년도/학회: 2024-10 / EMNLP 2024 Findings
- Problem: LLM과의 상호작용에서 시스템 프롬프트에 "You are a helpful assistant"와 같은 페르소나를 추가하는 것이 일반적입니다. 그러나 이러한 페르소나가 모델의 객관적 작업 수행 능력에 어떤 영향을 미치는지는 명확하지 않습니다.
- Takeaway
    - 162개의 다양한 사회적 역할(페르소나)을 선정하여 LLM의 성능에 미치는 영향을 체계적으로 평가했습니다.
    - 4개의 주요 LLM 계열과 2,410개의 MMLU dataset을 사용하여 실험을 수행했습니다. (metric: Accuracy)
    - 페르소나를 시스템 프롬프트에 추가하는 것이 모델 성능을 향상시키지 않음을 발견했습니다.
    - 페르소나의 성별, 유형, 도메인이 예측 정확도에 영향을 줄 수 있음을 추가 분석을 통해 제시했습니다.
- Method
- Experiments
    - Dataset
    - Model
    - Baselines
    - 페르소나를 추가하는 것이 모델의 성능을 향상시키지 않으며, 일부 경우에는 오히려 성능이 저하되었습니다.
    - 성별 중립적이고 도메인과 일치하는 업무 관련 역할이 더 나은 성능을 보였지만, 효과는 미미하였습니다.
    - 페르소나의 단어 빈도, 프롬프트-질문 간 유사성, 퍼플렉서티 등의 요소가 성능 차이를 설명하는 데 제한적이었습니다
- Result
- Limitations
- Insight
    - 단순한 "You are a/an {}"인 system prompt는 성능을 향상시키지 않음을 실험적으로 보여줌.
    - 보다 정교한 페르소나 설계 및 적용 방법이 필요함을 시사함.

### Persona is a Double-edged Sword: Mitigating the Negative Impact of Role-playing Prompts in Zero-shot Reasoning Tasks
- link: https://arxiv.org/abs/2408.08631
- 저자: Junseok Kim, Nakyeong Yang, Kyomin Jung 
- 년도/학회: 2024-10 / arXiv
- Problem: 최근 연구들은 LLM에 특정 페르소나(예: "당신은 유능한 수학 교사입니다")를 부여하면 추론 능력이 향상될 수 있음을 보여주었습니다. 그러나 이러한 페르소나가 항상 긍정적인 효과를 주는 것은 아니며, 부정확하게 정의된 페르소나는 오히려 모델의 추론 능력을 저하시킬 수 있습니다.
- Takeaway
    - 페르소나 프롬프트가 LLM의 추론 능력을 저해할 수 있음을 실험적으로 입증했습니다.
    - 역할 기반 프롬프트와 중립 프롬프트의 결과를 결합하여 추론 능력을 향상시키는 새로운 프레임워크인 Jekyll & Hyde를 제안했습니다.
    - LLM이 생성한 페르소나가 수작업으로 작성된 페르소나보다 더 안정적인 결과를 제공함을 발견했습니다.
- Method: Jekyll & Hyde Framework
    1. Persona Generator: LLM을 사용하여 주어진 질문에 적합한 페르소나를 자동으로 생성합니다.
    2. Dual Solvers: Persona Solver & Neutral Sovler
    3. Evaluator: 두 솔버의 답변을 비교하여 더 나은 답변을 선택합니다. 이 과정에서 응답 순서에 따른 위치 편향(position bias)을 완화하기 위해 다양한 순서로 평가를 반복합니다.
- Experiments
    - Dataset
        - Arithmatic: Multi-Arith, GSM8K, AddSub, AQUA-RAT, SingleEq, SVAMP
        - Commonsense reasoning: CSQA, StrategyQA
        - Symbolic reasoning: Last Letter Concatenation, Coin Flip
        - Others: Data Understanding, Tracking Shuffled Objects
    - Model: GPT-4 (gpt-4-0613), GPT-3.5-turbo (gpt-3.5-turbo-0125), llama3
    - Baselines
        - Base: only Neutral solver
        - Persona: only Persona solver
- Results
    - Persona prompt는 12개 중 7개의 dataset에서 LLM의 추론 능력을 저하시킴
    - Jekyll & Hyde Framework는 모든 dataset에서 추론 능력을 향상시킴
    - LLM이 생성한 persona는 hand-crafted인 persona보다 일관된 성능을 보임
- Limitations: Jekyll & Hyde 프레임워크는 각 질문에 대해 두 번의 추론과 평가를 수행하므로 계산 비용이 증가합니다.
- Insight: Persona prompt가 항상 LLM의 성능을 향상시키지 않으며, 오히려 저해할 수 있음을 실험적으로 보여줌. Jekyll & Hyde Framework는 이를 완화하는 방법론임. 

### Better Zero-Shot Reasoning with Role-Play Prompting
- link: https://aclanthology.org/2024.naacl-long.228/
- Github: https://github.com/NKU-HLT/Role-Play-Prompting
- 저자: Aobo Kong, Shiwan Zhao, Hao Chen, Qicheng Li, Yong Qin, Ruiqi Sun, Xin Zhou, Enzhi Wang, Xiaohang Dong
- 년도/학회: 2024-03 / NAACL 
- Problem: 대형 언어 모델(LLM)은 다양한 역할을 수행하는 능력을 갖추고 있으며, 이는 사용자와의 상호작용을 풍부하게 합니다. 그러나 이러한 역할 수행이 LLM의 추론 능력에 어떤 영향을 미치는지는 충분히 탐구되지 않았습니다. 본 논문은 전략적으로 설계된 역할 기반 프롬프트(role-play prompting)를 도입하여, 제로샷(zero-shot) 설정에서 다양한 추론 벤치마크에 대한 성능을 평가합니다.
- Takeaway
    - 역할 기반 프롬프트를 도입하여 LLM의 제로샷 추론 성능을 향상시킴.
    - 12개의 다양한 추론 벤치마크에서 역할 기반 프롬프트의 효과를 실증적으로 평가함.
    - 역할 기반 프롬프트가 Chain-of-Thought(CoT) 추론을 유도하는 더 효과적인 트리거로 작용함을 보여줌.
- Method
    - Role-Setting Prompt
    - Role-Feedback Prompt
- Experiment
    - Dataset
        - Arithmatic: Multi-Arith, GSM8K, AddSub, AQUA-RAT, SingleEq, SVAMP
        - Commonsense reasoning: CSQA, StrategyQA
        - Symbolic reasoning: Last Letter Concatenation, Coin Flip
        - Others: Data Understanding, Tracking Shuffled Objects
    - Model: ChatGPT (gpt-3.5-turbo-0613)
    - Baselines: Zero-Shot-CoT, Few-Shot-CoT
- Result
    - 주요 성능 수치:
        - AQuA 데이터셋에서 정확도가 53.5%에서 63.8%로 향상
        - Last Letter Concatenation에서 정확도가 23.8%에서 84.2%로 향상
    - 주요 결과 요약: 역할 기반 프롬프트는 대부분의 데이터셋에서 표준 제로샷 프롬프트보다 우수한 성능을 보였으며, Zero-Shot-CoT보다도 더 효과적인 CoT 유도 방법으로 나타남.
    - 부가 실험/분석: 역할 기반 프롬프트가 CoT 추론을 자연스럽게 유도하여, 모델의 추론 능력을 향상시킴
- Limitations
    - 특정 역할이 특정 작업에 더 적합할 수 있으므로, 역할 설정의 일반화에 대한 추가 연구가 필요함.
    - 역할 기반 프롬프트의 설계가 수작업에 의존하므로, 자동화된 역할 생성 방법에 대한 연구가 요구됨.
- Insight
    - ??

### ExpertPrompting: Instructing Large Language Models to be Distinguished Experts
- link: https://arxiv.org/pdf/2305.14688
- 저자: Benfeng Xu, An Yang, Junyang Lin, Quan Wang, Chang Zhou, Yongdong Zhang, Zhendong Mao 
- 년도/학회: 2025-03 / arXiv
- Problem
    - 이 논문은 어떤 문제를 해결하려고 하는가? 대형 언어 모델(LLM)의 응답 품질은 프롬프트 설계에 크게 의존한다. 이 논문은 LLM이 특정 도메인에서 전문가처럼 답변하도록 유도하는 프롬프트 전략을 제안하여, 일반적인 응답보다 더 정확하고 상세한 답변을 생성하는 문제를 해결하려 한다.
    - 이 문제는 왜 중요한가? LLM은 다양한 작업에서 활용되지만, 전문 지식이 필요한 질문에 대해 일반적인 답변을 제공하는 경우가 많다. 전문가 수준의 응답을 생성할 수 있다면, 교육, 고객 서비스, 의료 등 여러 분야에서 LLM의 실용성과 신뢰성이 크게 향상될 수 있다.
- Takeaway
    - Expert Prompting 제안: In-Context Learning을 활용해 각 명령에 맞는 전문가 정체성(expert identity)을 자동으로 생성하는 새로운 프롬프트 전략을 개발하여 LLM의 응답 품질을 향상시켰다.
    - ExpertLlaMa 개발: ExpertPrompting을 적용해 GPT-3.5로 생성한 고품질 명령-응답 데이터를 기반으로 오픈소스 챗 어시스턴트 ExpertLLaMA를 훈련시켰다.
    - 성능 향상 입증: GPT-4 기반 평가를 통해 ExpertLLaMA가 기존 오픈소스 모델(Alpaca, Vicuna 등)을 능가하며 원래 ChatGPT의 96% 수준에 달하는 성능을 달성함을 보였다.
- Method
    - 전체 구조: ExpertPrompting은 특정 명령에 맞는 전문가 정체성을 자동 생성한 후, 이를 기반으로 LLM이 전문가처럼 답변하도록 유도한다. 생성된 데이터를 활용해 ExpertLLaMA를 훈련시킨다.
    - 주요 구성 요소:   
        1. 전문가 정체성 생성: In-Context Learning을 통해 명령별로 맞춤화된 전문가 설명(예: 영양사, 물리학자)을 생성.
        2. 프롬프트 증강: 전문가 정체성을 명령에 추가해 LLM(GPT-3.5-turbo)에 입력.
        3. 데이터 생성 및 훈련: 52k Alpaca 명령어를 기반으로 고품질 데이터를 생성하고, 이를 LLaMA에 적용해 ExpertLLaMA를 훈련.
- Experiment
    - Dataset: 52k Alpaca dataset을 사용해 ExpertPrompting으로 새로운 QA data를 생성. 평가용으로 500개 랜덤 샘플링으로 추출. 
    - Evaluation: GPT-4를 활용한 automated evaluation으로 Expert 응답과 일반 응답의 품질 비교를 수행함. ExpertLlaMa의 성능은 ChatGPT 대비 상대 점수로 측정
    - Baselines: Alpaca, Vicuna, LlaMA-GPT4
- Result
    - 주요 성능 수치:
        - ExpertPrompting으로 생성한 데이터는 일반 응답보다 "상당히 높은 품질"을 보임 (GPT-4 평가).
        - ExpertLLaMA는 ChatGPT의 96% 수준 성능을 달성하며, Alpaca, Vicuna, LLaMA-GPT4를 능가.
    - 주요 결과 요약:
        - ExpertPrompting은 도메인별 전문성을 반영한 응답을 생성해 LLM의 성능을 크게 개선.
        - ExpertLLaMA는 오픈소스 모델 중 가장 경쟁력 있는 성능을 보여 ChatGPT에 근접.
    - 부가 실험/분석: 500개 샘플에 대한 GPT-4 평가에서 ExpertPrompting 응답이 일반 응답보다 더 상세하고 정확함을 확인.
- Limitations
    - 한계:
        - 논문에서 명시적 한계는 언급되지 않았으나, ExpertPrompting은 전문가 정체성 생성 과정의 확장성과 효율성 문제(수동 생성 비현실적)가 잠재적 한계로 추론됨.
        - 데이터셋 크기가 52k로 제한적이며, 더 큰 규모의 데이터로 훈련 시 성능 향상 가능성. 
    - 향후 연구 방향:
        - 더 큰 규모의 명령 데이터를 활용해 ExpertLLaMA 성능 개선.
        - 다양한 LLM에 ExpertPrompting 적용을 통해 일반화 가능성 탐구.
        - 전문가 정체성 생성의 효율성을 높이는 자동화 기법 개발.
- Insight
    - 전체적으로 느낀 점: ExpertPrompting은 LLM의 응답 품질을 전문가 수준으로 끌어올리는 간단하면서도 효과적인 방법으로, 프롬프트 엔지니어링의 중요성을 다시금 강조한다. ExpertLLaMA의 오픈소스 접근성은 연구자와 개발자에게 큰 가치를 제공한다.
    - 인상 깊은 부분: In-Context Learning을 활용한 자동화된 전문가 정체성 생성 아이디어가 실용적이고 확장 가능성이 높다. ChatGPT의 96% 성능을 오픈소스 모델로 달성한 점은 비용 효율적 대안으로서 큰 잠재력을 보여준다.
    - 의문점: 전문가 정체성의 품질이 응답에 미치는 구체적 영향(예: 정체성의 세부 수준 vs. 성능)과, 다양한 도메인에서의 일반화 가능성에 대한 추가 분석이 필요해 보인다. 또한, GPT-4 평가의 객관성에 대한 의문이 남는다.

## Summary & Insight
1. Persona prompting은 open-ended tasks에 효과적이다.
2. Persona prompting은 accuracy-based tasks (ex. classification)에 효과적이지 않다.
3. 만약 persona를 사용한다면, specific (not simple), detailed (give more contexts), ideally automated (not hand-crafted)되어야 한다. 

※ open-ended tasks
- 정의: 정해진 정답이 없고 다양한 방식의 응답이 가능한 task를 의미함.
- 예시: text generation, summarization, question & answering, code generation