# Role-Prompting: Does Adding Personas to Your Prompts Really Make a Difference?

본 글은 [Role-Prompting: Does Adding Personas to Your Prompts Really Make a Difference?](https://www.prompthub.us/blog/role-prompting-does-adding-personas-to-your-prompts-really-make-a-difference) 을 참조하여 4가지 논문에 대한 정리글입니다. 


## What is Role-Prompting (a.k.a Persona Prompting)?
LLM에게 특정 역할(ex. scientist)을 부여하여 그 역할에 맞는 어휘, 지식 수준, 시각으로 응답하도록 유도하는 기법이다. 이를 통해 모델의 응답이 보다 일관되고 특정 도메인에 맞는 정보 제공이 가능하다. 

## Research

### When "A Helpful Assistant" Is Not Really Helpful: Personas in System Prompts Do Not Improve Performances of Large Language Models
- link: https://arxiv.org/pdf/2311.10054
- 저자: Mingqian Zheng, Jiaxin Pei, Lajanugen Logeswaran, Moontae Lee, David Jurgens
- 년도/학회: 2024-10 / EMNLP 2024 Findings (17회 인용)
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
- 년도/학회: 2024-10 / arXiv (2회 인용)
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
    - Comparison
        - Base: only Neutral solver (w/o persona prompt)
        - Persona: only Persona solver (w/ persona prompt)
- Results
    - Persona prompt는 12개 중 7개의 dataset에서 LLM의 추론 능력을 저하시킴
    - Jekyll & Hyde Framework는 모든 dataset에서 추론 능력을 향상시킴
    - LLM이 생성한 persona는 hand-crafted인 persona보다 일관된 성능을 보임
- Limitations: Jekyll & Hyde 프레임워크는 각 질문에 대해 두 번의 추론과 평가를 수행하므로 계산 비용이 증가합니다.
- Insight: Persona prompt가 항상 LLM의 성능을 향상시키지 않으며, 오히려 저해할 수 있음을 실험적으로 보여줌. Jekyll & Hyde Framework는 이를 완화하는 방법론임. 

### Better Zero-Shot Reasoning with Role-Play Prompting (2024 NAACL, 173회 인용)
- link: https://aclanthology.org/2024.naacl-long.228/
- Github: https://github.com/NKU-HLT/Role-Play-Prompting
- 저자: Aobo Kong, Shiwan Zhao, Hao Chen, Qicheng Li, Yong Qin, Ruiqi Sun, Xin Zhou, Enzhi Wang, Xiaohang Dong
- 년도/학회: 2024-03 / NAACL 
- Problem: LLM (ex. GPT-3, PaLM, Llama)은 promptings을 통해 다양한 taks에서 뛰어난 성능을 보여줌 (때론 fine-tuning보다 우수함). 대표적으로 role prompting (i.e., persona prompting) 기법이 존재함. 그러나 이러한 role-prompting이 LLM의 reasoning ability에 어떤 영향을 미치는지는 충분히 탐구되지 않았음. 본 논문은 Role-play prompting를 도입하여, 제로샷(zero-shot) 설정에서 다양한 reasoning 벤치마크에 대한 성능을 평가 및 분석함.
- Takeaway
    - Role-Play Prompting (Role-Setting Prompt, Role-Feedback Prompt)를 도입하여 LLM의 제로샷 상태에서 Reasoning 성능을 향상시킴.
    - 12개의 다양한 Reasoning 벤치마크에서 Role-Play Prompting의 효과를 실증적으로 평가함.
    - Role-Play Prompting가 Chain-of-Thought(CoT) 보다 더 효과적인 트리거로 작용함을 보여줌.
- Method
    - 1단계) Role-Setting Prompt
        - task-specific role-play prompt를 구축. Role-Setting prompt를 디자인할 때, 1) 특정 task에 대해 차별화된 장점을 지닌 role을 선택 (Table 5)하고 2) 최대한 role에 대해 상세히 설명하는 것이 중요함 (Table 4).
        - 예: From now on, you are a contestant in the general knowledge quiz contest adn always answer all kinds of common sense questions accurately. I am the moderator of the game and the final is about to start.
    - 2단계) Role-Feedback Prompt
        - 만들어진 role-play prompt를 바탕으로 (예상되는) model의 response를 추가. 본 논문에서는 다수 후보군을 생성한 뒤, 성능이 가장 높이 나온 것으로 선택함. 
        - 예: That sounds like an exciting challenge! I'm ready to participate in the quiz contest as a contestant. Please go ahead and start the final round - I'm here to provide accurate answers to your commmon sense questions.
- Experiment
    - Dataset (metric: Accuracy)
        - Arithmatic (산술): Multi-Arith, GSM8K, AddSub, AQUA-RAT, SingleEq, SVAMP
        - Commonsense reasoning (일반 상식 추론): CSQA, StrategyQA
        - Symbolic reasoning (기호 추론): Last Letter Concatenation, Coin Flip
        - Others: Data Understanding, Tracking Shuffled Objects
    - Model: ChatGPT (gpt-3.5-turbo-0613)
    - Hyperpameters: greedy decoding (temp=0)
    - Comparison:
        - zero-shot prompting: user query w/o any additional prompt  
        - [Zero-Shot CoT](https://arxiv.org/abs/2205.11916) (2022 NeurIPS, 4961회 인용): user query w/ "Let's think step by step"
        - [Few-Shot CoT](https://arxiv.org/abs/2201.11903) (2022 NeurIPS, 14066회 인용): user query w/ "Let's think step by step" + similar examples (question-reasoning processes-answer)
- Result
    - 주요 성능 수치:
        - AQuA 데이터셋에서 accuarcy가 53.5% (zero-shot prompting) 대비 63.8%로 향상
        - Last Letter 데이터셋에서 accuracy가 23.8% (zero-shot prompting) 대비 84.2%로 향상
    - 주요 결과 요약:
        - Role-Play Prompting은 대부분의 데이터셋에서 zero-shot prompting, zero-shot CoT보다 우수한 성능을 보임 (outperforming 10 out of 12, 9 out of 12)
        - Few-Shot CoT과는 거의 tie함 (outperforming 6 out of 12)
    - 부가 실험/분석: Role-Play Prompting가 단계적으로 reasoning process를 강화함을 보여줌 (Table 3) 
- Limitations
    - 특정 role이 특정 task에 더 적합할 수 있으므로, 역할 설정의 일반화에 대한 추가 연구가 필요함.
    - 역할 기반 프롬프트의 설계가 수작업(hand-crafted)에 의존하므로, 자동화된(automated) 역할 생성 방법에 대한 연구가 요구됨.
- Insight
    - Role-Setting Prompt를 구축하는 방법에서 아이디어를 얻을 수 있었음. (Tip 2가지)
    - zero-shot prompting/CoT 대비 전반적으로 우수한 성능한 확인할 수 있었지만 Few-shot CoT에서는 그렇지 않음. (경쟁력있는 성능 X)

### ExpertPrompting: Instructing Large Language Models to be Distinguished Experts (2023, 130회 인용)
- link: https://arxiv.org/pdf/2305.14688
- 저자: Benfeng Xu, An Yang, Junyang Lin, Quan Wang, Chang Zhou, Yongdong Zhang, Zhendong Mao 
- 년도/학회: 2023-03 (처음 공개) => 2025-05 (Updated) / arXiv
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
3. 만약 persona prompting를 적용한다면, specific (unambiguous), detailed (give more contexts), ideally automated (not hand-crafted)되어야 한다. 

※ open-ended tasks
- 정의: 정해진 정답이 없고 다양한 방식의 정답이 가능한 task를 의미함.
- 예시: text generation, summarization, code generation
