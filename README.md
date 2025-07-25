# 📚 aiEducation
ML/DL, NLP 관련 공부 기록

* 기간 : 2021 ~ Present
* [석사 1년차에서의 회고록](https://velog.io/@jeewoo1025/%EC%84%9D%EC%82%AC-1%EB%85%84%EC%B0%A8%EC%97%90%EC%84%9C%EC%9D%98-%ED%9A%8C%EA%B3%A0%EB%A1%9D)
* [NLP 관련 논문리뷰 및 기본개념 설명](https://velog.io/@jeewoo1025/series/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0)
* [논문 초안 작성부터 결과 발표까지의 Process](https://velog.io/@jeewoo1025/%EB%85%BC%EB%AC%B8-%EC%B4%88%EC%95%88-%EC%9E%91%EC%84%B1%EB%B6%80%ED%84%B0-%EA%B2%B0%EA%B3%BC-%EB%B0%9C%ED%91%9C%EA%B9%8C%EC%A7%80%EC%9D%98-Process)
* [LLMs vs Agents 비교](https://jeewoo1025.tistory.com/6)
<br><br>

## 📢 Note
- `2025-06-15`: Code 폴더 업로드
- `2025-03-29`: All_about_Agent/Agent.md 업로드
<br><br>

## 최근 AI Trends Follow Up Tips
최근 LLMs 동향이 빠르게 바뀌고 있어 아래 리스트를 매일 참조하는 걸 추천한다. 
가장 추천하는 건 LinkedIn 가입 후 관련 포스트들을 Follow Up 하는 것이다. 국내외 연구자들의 의견들을 공유받을 수 있어 AI 연구자라면 필수!
최근 가장 흥미롭게 읽었던 글은 [조경현 교수님 - i sensed anxiety and frustration at NeurIPS'24](https://kyunghyuncho.me/i-sensed-anxiety-and-frustration-at-neurips24/)인데 석사 졸업 후 회사에서 최근 느꼈던 실제 산업과 학계에서의 gap에 대하여 잘 정리된 글이라고 생각되었다.
* [OpenAI Youtube](https://www.youtube.com/OpenAI)
* [조코딩 Youtube](https://www.youtube.com/@jocoding)
* [Deepseek Site](https://www.deepseek.com/)
<br><br>

## 🔎 논문 찾는 Tips
1. [paperswithcode](https://paperswithcode.com/sota)에서 tasks 위주로 SOTA 논문을 보여줌 <br> 
   ✔ Most implemented paper를 참고하면 어떤 논문이 가장 많이 인용되었는지 확인가능함 <br><br> 
2. github에서 task 검색 <br> 이때 `awesome [특정 task]`로 검색하면 curated list 게시물을 쉽게 찾을 수 있다. <br> 얼마나 중요한 논문인지는 Star 갯수나 fork 수로 판별가능함
   <br> <br>

3. ACL, EMNLP, NAACL 등 ACL 계열 학회 같이 h5-index가 높은 학회들(Conference)에서 발표한 논문들로 최신 트렌드를 알 수 있다. 
* [AI h5-index](https://scholar.google.es/citations?view_op=top_venues&hl=en&vq=eng_artificialintelligence), [NLP h5-index](https://scholar.google.es/citations?view_op=top_venues&hl=en&vq=eng_computationallinguistics)
<br><br>

## 🍃 논문 작성법
### Overleaf
Latex는 Conference, Journal 등 논문을 작성할 수 있도록 도와주는 문서 작성 시스템이다. 대다수의 논문들이 Latext를 이용해 작성되고, 공유되어 관리되어 있다. 
이러한 Latex 프로그램을 사용해 논문 프로젝트를 편하게 관리하고 공유할 수 있도록 해주는 대표적인 서비스로 `Overleaf`가 있다. 
`Overleaf > Template`에서 검색을 통해 제출할 학회의 논문 Template를 다운받아 작성하면 된다. 
* 사이트 : https://www.overleaf.com/project
* 사용법 : [나동빈 > 이공계열 학생을 위한 Latex 작성 방법 Feat. Overleaf](https://ndb796.tistory.com/342)

▶ LaTex 기호 정리 : https://jjycjnmath.tistory.com/117
<br><br>

## 🤖 LLMs
### Deepseek R1
- `2025-01-22` Deepseek-R1: https://github.com/deepseek-ai/DeepSeek-R1
- [Building DeepSeek R1 from Scratch](https://github.com/FareedKhan-dev/train-deepseek-r1)
- [DeepSeek-R1에 관한 비주얼 가이드](https://tulip-phalange-a1e.notion.site/DeepSeek-R1-189c32470be2801c94b6e5648735447d)

### Llama-Factory
- Paper: [LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models](https://arxiv.org/abs/2403.13372) (ACL 2024)
- Gtihub: https://github.com/hiyouga/LLaMA-Factory
- `How to use in Korean`: [아이언맨 페르소나 Fine tuning 해보기](https://pseudorec.com/archive/monthly_pseudorec/12/)

### Agents
- [LLM Agents에 관한 비주얼 가이드](https://tulip-phalange-a1e.notion.site/LLM-Agents-1b9c32470be2800fa672e82689018fc4)
- `Multi-Agent에서 가장 영향력 있는 논문` [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) (`2023`, 1903회 인용)
- [챗GPT와 LangChain을 활용한 LLM 기반 AI 서비스 앱 개발](https://github.com/ychoi-kr/langchain-book)
- [Awesome Langchain](https://github.com/kyrolabs/awesome-langchain)

### OpenAI
- [Function Calling 기능](https://platform.openai.com/docs/guides/function-calling?api-mode=responses)

### vLLM 
- Github: https://github.com/vllm-project/vllm
- [vLLM 사용법 - LLM 쉽게 빠르게 추론 및 서빙하기](https://lsjsj92.tistory.com/668)

### Prompt Engineering
- [Prompt Engineering Tutorial](https://github.com/NirDiamant/Prompt_Engineering)
- [Prompt Engineering Guide](https://www.promptingguide.ai/kr)

<br><br>

## 🦜 Reasoning
- [Reasoning LLMs에 관한 비주얼 가이드](https://tulip-phalange-a1e.notion.site/Reasoning-LLMs-190c32470be2806d834ee0ad98aaa0b6)

<br><br>

## ⭐ NLP 필수 논문 (년도 순)
<b>논문 년도 순서별로 읽는 걸 추천한다. </b> 왜냐하면, 이전 년도의 나온 논문들을 이해해야 현재 논문을 이해할 수 있기 때문! 
예를들어, MASS paper를 알아야 BART paper를 정확히 이해할 수 있다.
또한 유명한 논문들은 대부분 인용되어서 paper에 추가됨. 대표적인 예시) BERT paper에서 GPT와의 비교를 수행. 모두 2019년도에 publish됨. <br>
😀 아래의 models에 대한 공부하기 좋은 PyTorch code : https://github.com/paul-hyun/transformer-evolution
LLM 관련 최신 paper list를 찾아보려면 [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM?tab=readme-ov-file)을 참조하는 걸 추천한다.

* Word2Vec (ICLR 2013) : [Paper Link](https://arxiv.org/abs/1301.3781)
* Seq2Seq (NIPS 2014) : [Paper Link](https://arxiv.org/abs/1409.3215) / [Seq2Seq.pdf](https://github.com/jeewoo1025/aiEducation/files/7446757/Seq2Seq.pdf) / [colab](https://colab.research.google.com/drive/1Jg4AYB-Ku4tuSIRchU8REvwqlsfsPD81#scrollTo=1OSgbkh0Vkq7)
* bahdanau Attetion (ICLR 2015) : [Paper Link](https://arxiv.org/abs/1409.0473) 
* Transformer (NIPS 2017) : [Paper Link](https://arxiv.org/abs/1706.03762v5)
    * [Pytorch tutorial Harvard's NLP group](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
* GPT (2018) : [Paper Link](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) 
* BERT (NACCL 2019) : [Paper Link](https://arxiv.org/abs/1910.13461v1) 
* GPT-2 (2019) : [Paper Link](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* RoBERTa (ICLR 2019) : [Paper Link](https://arxiv.org/abs/1907.11692)
* GPT-3 (NIPS 2020) : [Paper Link](https://arxiv.org/abs/2005.14165) 
* BART (ACL 2020) : [Paper Link](https://arxiv.org/abs/1910.13461)
   * huggingface bart : https://huggingface.co/transformers/v2.11.0/model_doc/bart.html

### ✔ Summary
||Base model|Pretraining Tasks|
|:---:|:---:|:---:|
|ELMo|two-layer biLSTM|next token prediction|
|GPT|Transformer decoder|next token prediction|
|BERT|Transformer encoder|mask language model + next sentence prediction|
|ALBERT|same as BERT but light-weighted|mask language model + sentence order prediction|
|GPT-2|Transformer decoder|next token prediction|
|RoBERTa|same as BERT|mask language model (dynamic masking)|
|T5|Transformer encoder + decoder|pre-trained on a multi-task mixture of unsupervised and supervised tasks and for which each task is converted into a text-to-text format.|
|GPT-3|Transformer decoder|next token prediction|
|BART|BERT encoder + GPT decoder|reconstruct text from a noised version|
|ELECTRA|same as BERT|replace token detection|
|LoRA|최근 학습 시 필수적으로 사용하는 기법|
|CoT|CoT 개념 잡기에 좋음|
|FLAN|Instruction-finetuning 처음으로 제시함.|
<br>

## Code 추천 논문
Code LLM 관련 최신 논문들은 [Awesome-Code-LLM](https://github.com/codefuse-ai/Awesome-Code-LLM)을 참조하는 걸 추천한다.
<br><br>

## Text Summarization 추천 논문
### Baseline model
baseline model로 BART가 주로 쓰이긴 한다. 하지만 Pegasus model도 XSum dataset에서 많이 쓰인다. 
* BART (ACL, 2020)
* PEGASUS : Pre-training with Extracted Gap-sentences for Abstractive Summarization (ICML, 2020)
<br>

### Abstractive Summarization
논문을 읽을 때 short와 long paper를 구분해서 읽기 바란다. long과 short paper 각각의 contribution이 크게 차이가 나기 때문. 
* Abstractive Text Summarization using Sequence-to-Sequence RNNs and beyond (CONLL, 2016)
* Text Summarization with Pretrained Encoders (EMNLP, 2019)
* RefSum: Refactoring Neural Summarization (NAACL, 2019)
* SimCLS: A Simple Framework for Contrastive Learning of Abstractive Summarization (ACL short, 2021)
* SummaReranker: A Multi-Task Mixture-of-Experts Re-ranking Framework for Abstractive Summarization (ACL, 2022)
* BRIO: Bringing Order to Abstractive Summarization (ACL, 2022)
<br>

### Extractive Summarization
* Extractive Summarization as Text Matching (ACL, 2020)
* GSum: General Framework for Guided Neural Summarization (NAACL, 2021)
<br>

#### Contrastive Learning 추천 논문
원래 Computer Vision에서 처음 소개된 기법이기 때문에 비전쪽 논문도 읽는 것을 추천함

#### Computer Vision
* A Simple Framework for Contrastive Learning of Visual Representations (ICML, 2020)
* Understanding contrastive representation learning through alignment and uniformity on the hypersphere (ICML, 2020) (처음으로 contrastive learning의 잘되는 핵심적인 이유인 alignment과 uniformity analysis를 제시함.)

### NLP
* SimCSE: Simple Contrastive Learning of Sentence Embeddings (EMNLP, 2021)
* Debiased Contrastive Learning of Unsupervised Sentence Representations (ACL, 2022)
* A Contrastive Framework for Learning Sentence Representations from Pairwise and Triple-wise Perspective in Angular Space (ACL, 2022)
<br>

## 📊 성능 측정 방법
1. **BLEU** 
* Bilingual Evaluation Understudy
* 기계번역의 성능이 얼마나 뛰어난가를 측정하기 위해 사용함
* 기계 번역 결과와 사람이 직접 번역한 결과가 얼마나 유사한지 비교하여 번역에 대한 성능을 측정하는 방법
* 높을 수록 성능이 더 좋다
* 장점 : 언어에 구애받지 않음, 계산 속도가 빠름
<br>
<br>

2. **ROUGE / ROUGE 2.0**
* Recall-Oriented Understudy for Gisting Evaluation
* github : https://github.com/bheinzerling/pyrouge
* Text summarization의 성능을 측정하기 위해 사용함
* ROUGE는 reference summary와 모델이 생성한 summary 사이에 겹치는 token이 많을수록 score가 높아진다. 하지만, 다른 단어라도 동일한 의미를 가지는 문장을 포함하지 않는다는 한계점이 있어서 이를 보완해서 나온게 ROUGE 2.0이다.
* ROUGE 2.0은 synonymous(동의어)와 topic coverage를 포함하여 위의 issue를 보완하였다. → `ROUGE - {NN | Topic | TopicUniq} + Synonyms`
* 그러나 여전히 완벽하게 score 매길 수 없지만 현재까지 가장 좋은 Evaluation 방법이라고 평가받는다.
<br>
<br>

## 📬 투고 
* **Workshop** 
  * 대규모 학회는 시작할 때 앞뒤로 하루 규모의 workshop를 진행한다. 목적은 본 학회 참석자들이 specific한 키워드를 중심으로 모여서 진행하는 작은 학회같은 느낌. 보통 본 학회 내기 애매하거나 Working in Process를 공유하고 피드백 받는 자리이기도 하다.
  * Call for workshop을 열어 committee가 pass/non pass 여부를 주고 다시 그 workshop에서 받을 논문에 대한 공고를 낸다. 
<br>

* **Tutorial** 
  * 새로운 논문을 제안하기보다는 급 부상한 새로운 주제에 대한 개론적인 강의를 하는 하루 규모의 세션 (e.g. ACL 2020 open-domain QA tutorial)
<br>

* **Main Conference**
  * 가장 중요한 메인 컨퍼런스이다. Accepted Papers의 저자들이 Oral 또는 Poster Session으로 Methods를 발표한다.
  * ACL 계열 학회들 (ACL, EMNLP, NAACL, EACL, COLING)은 long/short paper로 나눠서 투고한다. 학회마다 기대하는 long/short paper에 대한 스펙이 있기 때문에, call for paper를 참고하는 걸 추천한다. 통상적으로 short paper는 long paper에 비해 상당히 짧고 contribution이 더 작다고 판단된다. 
  * NAACL call for papers 2022 
    * Long paper : (8 pages) substantial, original, completed and unpublished work
    * Short paper : (4 pages) original and unpublished work
<br>
<br>

## Dataset Download
|Dataset|Domain|Train|Val|Test|Doc #Tokens|Sum #Tokens|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|XSum|News|204,045|11,332|11,334|437.21|23.87|
|CNN/DM|News|287,113|13,368|11,490|782.67|58.33|

### XSum
* hungging face link :  https://huggingface.co/datasets/xsum
* get dataset
```
import datasets

dataset = datasets.load_dataset("xsum")
```
<br>

### CNN/DM
* hugging face link : https://huggingface.co/datasets/cnn_dailymail
* get dataset
```
import datasets

dataset = datasets.load_dataset("cnndm", "3.0.0")  # dataset name, version
```
<br>

## 🔬 Library
### Spacy 
* link : https://spacy.io/usage/spacy-101#whats-spacy
* English 자연어처리를 위한 Python 오픈소스 라이브러리. 
* 지원 기능 : Tokenization, POS Tagging, Dependency Parsing, NER, Similarity ...
```python
import spacy
nlp = spacy.load('en_core_web_sm')
```
<br>

##  📝  Study
### ACL / EMNLP / NAACL
* [ACL 2023: Generating Text from Language Models (ACL 2023: Tutorials)](https://rycolab.io/classes/acl-2023-tutorial/)

### 나동빈
* [꼼꼼한 딥러닝 논문 리뷰와 코드 실습](https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice)
* [컴퓨터 공학과를 위한 최신 논문 찾아 읽는 방법 정리](https://www.youtube.com/watch?v=FPcdxHCxH_o)
<br><br>

### Deep Learning 
* [PyTorch로 시작하는 딥 러닝 입문](https://wikidocs.net/book/2788)
* [BERT 돌아보기](https://docs.likejazz.com/bert/)
<br><br>

### 기타 지식
* [Knowledge Distillation](https://baeseongsu.github.io/posts/knowledge-distillation/)

### NLP basic
* [oh! suz's NLP Cookbook](https://www.ohsuz.dev/nlp-cookbook)
* [Jeonsworld's NLP 관련 논문 리뷰 포스팅](https://jeonsworld.github.io/)
* [월간 자연어처리 - Facebook](https://www.facebook.com/monthly.nlp?hc_ref=ART_4x3Knm-Y_6Rw38lFMtWmKZ8SdL4fWSzm2I9CiaYwJAtFIHk9mP_T7mK69NC8V2A&fref=nf&__xts__[0]=68.ARD8SbISh91tv-3NTdye910Za6oW4Nkfc9S3jAAX3n9xWPQjLdTDJA9eCQh_J10Y3ROSXAR5k_zgzd7q77OEgRaN0yMMkp4XdSPzROUANUkOJajbcUBhbaPtD_riFOG2cAWkFIAJ35CE3XQvrYj4246-Ggebd06AhnUK_WuOr-nZFECcT_txc0ekAqJC_OEvZaGzcYr8CwWwjCCYO2cg3reKqV6CrF2unShmou5PdNlmFzpiNrmYlltICYZxFX-mQdn0eBXJkpxKBr_b_pD1LnBO2e0QcFI_cC6plzalWQ3RbB6daGM)
* [딥 러닝을 이용한 자연어 처리 입문(Tensoflow/Keras 사용)](https://wikidocs.net/book/2155)
* [Generalized Language Models](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html#gpt)

 ### 기타 Tools
 - 다이어그램 툴 [Excalidraw](https://excalidraw.com/)
