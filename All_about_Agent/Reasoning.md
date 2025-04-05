# Reasoning

본 글은 [Reasoning LLMs에 관한 비주얼 가이드](https://tulip-phalange-a1e.notion.site/Reasoning-LLMs-190c32470be2806d834ee0ad98aaa0b6)을 요약한 정리본이다. 참고로 원문은 [A Visual Guide to Reasoning LLMs](https://tulip-phalange-a1e.notion.site/Reasoning-LLMs-190c32470be2806d834ee0ad98aaa0b6)이다.


## Test-Time Computing
사전 학습에서 Model (# of parameters) 크기를 키우거나, Dataset (# of tokens) 크기를 증가하거나, Compute (# of FLOPS) 연산 시간을 늘리지 않는 대신 inference 시점에 "더 오래 생각"할 수 있도록 함.

LLMs이 최종 답을 생성하는 과정에서 더 많은 resource를 사용해야 하는 것!
More tokens = More computer = More time to Think = Better performance

### Search against Verifiers
Question → Generate thoughts and answers → Verify thoughts and/or 
answer quality → Choose best answer <br>

#### ORM & PRM 
1. Outcome Reward Models (ORM)
: 오직 결과만을 평가 
2. Process Reward Models (PRM)
: 결과를 도출하는 과정 (reasoning) 과정도 결과와 함께 평가


- Majority Voting: 여러 답변을 생성하게 한 뒤, 가장 많이 나온 답을 최종 답으로 선택 (self-consistency)
- Best-of-N samples: N개의 샘플을 생성 후, ORM or PRM을 사용해 각 답변을 평가 후 가장 높은 점수를 받은 답변을 선택함
- Beam search with process reward models: Beam search를 사용하여 여러 reasoning 단계를 sampling하고 각 단계를 PRM이 평가. 점수가 높은 top-3의 beam을 계속 추적함
- Monte Carlo Tree Search: Selection → Expand → Rollouts → Backprop

<br>

### Modifying Proposal Distribution
Fine-tuning (learning to reason before answering) <br>
completions/thoughts/tokens이 sampling되는 분포를 조정하는 것

#### Self-Taught Reasoner (STaR)
1. Generate reasoning + answer <br>

`Correct Answer 경우` <br>
2. Correct answer <br>
3. Generate training data <br>
4. Generate triplet training data (Question, Reasoning, Answer) <br>
5. Supervised Fine-tuning <br>

`Incorrect Answer 경우` <br>
2. Incorrect answer <br>
3. Provide hint <br>
4. Generate reasoning only (why is this answer correct?) <br>
5. Generate triplet training data <br>
6. Supervised fine-tuning <br>