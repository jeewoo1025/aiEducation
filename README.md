# 📚 aiEducation
ML/DL, NLP 관련 공부 기록

* 기간 : 2021 ~ Present
<br><br>

## 🔎 논문 찾는 Tips
1. [paperswithcode](https://paperswithcode.com/sota)에서 tasks 위주로 SOTA 논문을 보여줌 <br> 
   ✔ Most implemented paper를 참고하면 어떤 논문이 가장 많이 인용되었는지 확인가능함 <br><br> 
2. github에서 task 검색 <br> 이때 `awesome [특정 task]`로 검색하면 curated list 게시물을 쉽게 찾을 수 있다. <br> 얼마나 중요한 논문인지는 Star 갯수나 fork 수로 판별가능함
   <br> <br>

3. EMNLP, ACL 등 Impact Factor가 높은 학회들에서 발표한 논문들로 최신 트렌드를 알 수 있다
<br><br>

## 🍃 논문 작성법
### Overleaf
Latex는 Conference, Journal 등 논문을 작성할 수 있도록 도와주는 문서 작성 시스템이다. 대다수의 논문들이 Latext를 이용해 작성되고, 공유되어 관리되어 있다. 
이러한 Latex 프로그램을 사용해 논문 프로젝트를 편하게 관리하고 공유할 수 있도록 해주는 대표적인 서비스로 `Overleaf`가 있다. 
`Overleaf > Template`에서 검색을 통해 제출할 학회의 논문 Template를 다운받아 작성하면 된다. 
* 사이트 : https://www.overleaf.com/project
* 사용법 : [나동빈 > 이공계열 학생을 위한 Latex 작성 방법 Feat. Overleaf](https://ndb796.tistory.com/342)

<br><br>

## ⭐ NLP 필수 논문 (년도 순)
* Word2Vec (ICLR 2013) : [Paper Link](https://arxiv.org/abs/1301.3781)
* Seq2Seq (NIPS 2014) : [Paper Link](https://arxiv.org/abs/1409.3215) / [Seq2Seq.pdf](https://github.com/jeewoo1025/aiEducation/files/7446757/Seq2Seq.pdf) / [colab](https://colab.research.google.com/drive/1Jg4AYB-Ku4tuSIRchU8REvwqlsfsPD81#scrollTo=1OSgbkh0Vkq7)
* Attetion (ICLR 2015) : [Paper Link]() 
* Transformer (NIPS 2017) : [Paper Link](https://arxiv.org/abs/1706.03762v5)
    * [Pytorch tutorial Harvard's NLP group](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

* GPT (2018) : [Paper Link](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) 
* BERT (NACCL 2019) : [Paper Link](https://arxiv.org/abs/1910.13461v1) 
* GPT-2 (2019) : [Paper Link](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* GPT-3 (NIPS 2020) : [Paper Link](https://arxiv.org/abs/2005.14165) 
* BART (ACL 2020) : [Paper Link](https://arxiv.org/abs/1910.13461)
<br>
<br>

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

<br>
<br>

### ✔ Open QA
* Danqi Chen github : https://github.com/danqi
* [ACL2020 Tutorial: Open-Domain Question Answering](https://github.com/danqi/acl2020-openqa-tutorial) 
<br>
<br>

### ✔ 텍스트 요약 
* Must-read paper : https://github.com/jeewoo1025/Text-Summarization-Repo
* 강필성 교수님의 DSBA 연구실 자료 : [github](https://github.com/pilsung-kang/text-analytics) / [youtube](https://www.youtube.com/channel/UCPq01cgCcEwhXl7BvcwIQyg)
* mathsyouth의 curated list: [awesome-text-summarization](https://github.com/mathsyouth/awesome-text-summarization)

#### Dataset
* [CNN-Daily Mail](https://github.com/abisee/cnn-dailymail)
<br>
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
* **workshop** 
  * 대규모 학회는 시작할 때 앞뒤로 하루 규모의 workshop를 진행한다. 목적은 본 학회 참석자들이 specific한 키워드를 중심으로 모여서 진행하는 작은 학회같은 느낌. 보통 본 학회 내기 애매하거나 Working in Process를 공유하고 피드백 받는 자리이기도 하다.
  * Call for workshop을 열어 committee가 pass/non pass 여부를 주고 다시 그 workshop에서 받을 논문에 대한 공고를 낸다. 
<br>

* **tutorial** 
  * 새로운 논문을 제안하기보다는 급 부상한 새로운 주제에 대한 개론적인 강의를 하는 하루 규모의 세션 (e.g. ACL 2020 open-domain QA tutorial)
<br>

* **full/short paper**
  * 학회마다 기대하는 long/short paper에 대한 스펙이 있기 때문에, call for paper를 참고하는 걸 추천한다. 통상적으로 short paper는 long paper에 비해 상당히 짧고 long paper 투고가 더 인정 받는다. 
  * NAACL call for papers 2022 
    * Long paper : (8 pages) substantial, original, completed and unpublished work
    * Short paper : (4 pages) original and unpublished work
<br>
<br>

## Dataset Download
Origin link : https://github.com/ShichaoSun/ConAbsSum
### XSum
* hungging face link :  https://huggingface.co/datasets/xsum
* get dataset
```
wget https://cdn-datasets.huggingface.co/summarization/xsum.tar.gz
tar -xzvf xsum.tar.gz
```
<br>

### CNN/DM
* hugging face link : https://huggingface.co/datasets/cnn_dailymail
* get dataset
```
wget https://cdn-datasets.huggingface.co/summarization/pegasus_data/cnn_dailymail.tar.gz
tar -xzvf cnn_dailymail.tar.gz
mv cnn_dailymail/validation.source cnn_dailymail/val.source 
mv cnn_dailymail/validation.target cnn_dailymail/val.target 
```
<br>

##  📝  Study
### 나동빈
* [꼼꼼한 딥러닝 논문 리뷰와 코드 실습](https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice)
* [컴퓨터 공학과를 위한 최신 논문 찾아 읽는 방법 정리](https://www.youtube.com/watch?v=FPcdxHCxH_o)
<br><br>

### Deep Learning 
* [PyTorch로 시작하는 딥 러닝 입문](https://wikidocs.net/book/2788)
* [BERT 돌아보기](https://docs.likejazz.com/bert/)
<br><br>

### NLP basic
* [oh! suz's NLP Cookbook](https://www.ohsuz.dev/nlp-cookbook)
* [Jeonsworld's NLP 관련 논문 리뷰 포스팅](https://jeonsworld.github.io/)
* [월간 자연어처리 - Facebook](https://www.facebook.com/monthly.nlp?hc_ref=ART_4x3Knm-Y_6Rw38lFMtWmKZ8SdL4fWSzm2I9CiaYwJAtFIHk9mP_T7mK69NC8V2A&fref=nf&__xts__[0]=68.ARD8SbISh91tv-3NTdye910Za6oW4Nkfc9S3jAAX3n9xWPQjLdTDJA9eCQh_J10Y3ROSXAR5k_zgzd7q77OEgRaN0yMMkp4XdSPzROUANUkOJajbcUBhbaPtD_riFOG2cAWkFIAJ35CE3XQvrYj4246-Ggebd06AhnUK_WuOr-nZFECcT_txc0ekAqJC_OEvZaGzcYr8CwWwjCCYO2cg3reKqV6CrF2unShmou5PdNlmFzpiNrmYlltICYZxFX-mQdn0eBXJkpxKBr_b_pD1LnBO2e0QcFI_cC6plzalWQ3RbB6daGM)
* [딥 러닝을 이용한 자연어 처리 입문(Tensoflow/Keras 사용)](https://wikidocs.net/book/2155)
* [Generalized Language Models](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html#gpt)
