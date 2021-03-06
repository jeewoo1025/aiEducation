# ๐ aiEducation
ML/DL, NLP ๊ด๋ จ ๊ณต๋ถ ๊ธฐ๋ก

* ๊ธฐ๊ฐ : 2021 ~ Present
<br><br>

## ๐ ๋ผ๋ฌธ ์ฐพ๋ Tips
1. [paperswithcode](https://paperswithcode.com/sota)์์ tasks ์์ฃผ๋ก SOTA ๋ผ๋ฌธ์ ๋ณด์ฌ์ค <br> 
   โ Most implemented paper๋ฅผ ์ฐธ๊ณ ํ๋ฉด ์ด๋ค ๋ผ๋ฌธ์ด ๊ฐ์ฅ ๋ง์ด ์ธ์ฉ๋์๋์ง ํ์ธ๊ฐ๋ฅํจ <br><br> 
2. github์์ task ๊ฒ์ <br> ์ด๋ `awesome [ํน์  task]`๋ก ๊ฒ์ํ๋ฉด curated list ๊ฒ์๋ฌผ์ ์ฝ๊ฒ ์ฐพ์ ์ ์๋ค. <br> ์ผ๋ง๋ ์ค์ํ ๋ผ๋ฌธ์ธ์ง๋ Star ๊ฐฏ์๋ fork ์๋ก ํ๋ณ๊ฐ๋ฅํจ
   <br> <br>

3. EMNLP, ACL ๋ฑ Impact Factor๊ฐ ๋์ ํํ๋ค์์ ๋ฐํํ ๋ผ๋ฌธ๋ค๋ก ์ต์  ํธ๋ ๋๋ฅผ ์ ์ ์๋ค
<br><br>

## ๐ ๋ผ๋ฌธ ์์ฑ๋ฒ
### Overleaf
Latex๋ Conference, Journal ๋ฑ ๋ผ๋ฌธ์ ์์ฑํ  ์ ์๋๋ก ๋์์ฃผ๋ ๋ฌธ์ ์์ฑ ์์คํ์ด๋ค. ๋๋ค์์ ๋ผ๋ฌธ๋ค์ด Latext๋ฅผ ์ด์ฉํด ์์ฑ๋๊ณ , ๊ณต์ ๋์ด ๊ด๋ฆฌ๋์ด ์๋ค. 
์ด๋ฌํ Latex ํ๋ก๊ทธ๋จ์ ์ฌ์ฉํด ๋ผ๋ฌธ ํ๋ก์ ํธ๋ฅผ ํธํ๊ฒ ๊ด๋ฆฌํ๊ณ  ๊ณต์ ํ  ์ ์๋๋ก ํด์ฃผ๋ ๋ํ์ ์ธ ์๋น์ค๋ก `Overleaf`๊ฐ ์๋ค. 
`Overleaf > Template`์์ ๊ฒ์์ ํตํด ์ ์ถํ  ํํ์ ๋ผ๋ฌธ Template๋ฅผ ๋ค์ด๋ฐ์ ์์ฑํ๋ฉด ๋๋ค. 
* ์ฌ์ดํธ : https://www.overleaf.com/project
* ์ฌ์ฉ๋ฒ : [๋๋๋น > ์ด๊ณต๊ณ์ด ํ์์ ์ํ Latex ์์ฑ ๋ฐฉ๋ฒ Feat. Overleaf](https://ndb796.tistory.com/342)

โถ LaTex ๊ธฐํธ ์ ๋ฆฌ : https://jjycjnmath.tistory.com/117
<br><br>

## โญ NLP ํ์ ๋ผ๋ฌธ (๋๋ ์)
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
   * huggingface bart : https://huggingface.co/transformers/v2.11.0/model_doc/bart.html
<br>
<br>

### โ Summary
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

#### Huggingface/transformers document 
Huggingface์์ ์ ๊ณตํ๋ transformers document
* [generate](https://huggingface.co/docs/transformers/v4.19.2/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate)
* [BART](https://huggingface.co/docs/transformers/model_doc/bart)
* [Pegasus](https://huggingface.co/docs/transformers/model_doc/pegasus)
<br>
<br>

### โ Open QA
* Danqi Chen github : https://github.com/danqi
* [ACL2020 Tutorial: Open-Domain Question Answering](https://github.com/danqi/acl2020-openqa-tutorial) 
<br>
<br>

### โ ํ์คํธ ์์ฝ 
Summarization study ์ ๋ฆฌ > [๋งํฌ](https://github.com/jeewoo1025/aiEducation/blob/main/Summarization.md)
* Must-read paper : https://github.com/jeewoo1025/Text-Summarization-Repo
* ๊ฐํ์ฑ ๊ต์๋์ DSBA ์ฐ๊ตฌ์ค ์๋ฃ : [github](https://github.com/pilsung-kang/text-analytics) / [youtube](https://www.youtube.com/channel/UCPq01cgCcEwhXl7BvcwIQyg)
* mathsyouth์ curated list: [awesome-text-summarization](https://github.com/mathsyouth/awesome-text-summarization)

#### Dataset
* [CNN-Daily Mail](https://github.com/abisee/cnn-dailymail)
<br>
<br>

## ๐ ์ฑ๋ฅ ์ธก์  ๋ฐฉ๋ฒ
1. **BLEU** 
* Bilingual Evaluation Understudy
* ๊ธฐ๊ณ๋ฒ์ญ์ ์ฑ๋ฅ์ด ์ผ๋ง๋ ๋ฐ์ด๋๊ฐ๋ฅผ ์ธก์ ํ๊ธฐ ์ํด ์ฌ์ฉํจ
* ๊ธฐ๊ณ ๋ฒ์ญ ๊ฒฐ๊ณผ์ ์ฌ๋์ด ์ง์  ๋ฒ์ญํ ๊ฒฐ๊ณผ๊ฐ ์ผ๋ง๋ ์ ์ฌํ์ง ๋น๊ตํ์ฌ ๋ฒ์ญ์ ๋ํ ์ฑ๋ฅ์ ์ธก์ ํ๋ ๋ฐฉ๋ฒ
* ๋์ ์๋ก ์ฑ๋ฅ์ด ๋ ์ข๋ค
* ์ฅ์  : ์ธ์ด์ ๊ตฌ์ ๋ฐ์ง ์์, ๊ณ์ฐ ์๋๊ฐ ๋น ๋ฆ
<br>
<br>

2. **ROUGE / ROUGE 2.0**
* Recall-Oriented Understudy for Gisting Evaluation
* github : https://github.com/bheinzerling/pyrouge
* Text summarization์ ์ฑ๋ฅ์ ์ธก์ ํ๊ธฐ ์ํด ์ฌ์ฉํจ
* ROUGE๋ reference summary์ ๋ชจ๋ธ์ด ์์ฑํ summary ์ฌ์ด์ ๊ฒน์น๋ token์ด ๋ง์์๋ก score๊ฐ ๋์์ง๋ค. ํ์ง๋ง, ๋ค๋ฅธ ๋จ์ด๋ผ๋ ๋์ผํ ์๋ฏธ๋ฅผ ๊ฐ์ง๋ ๋ฌธ์ฅ์ ํฌํจํ์ง ์๋๋ค๋ ํ๊ณ์ ์ด ์์ด์ ์ด๋ฅผ ๋ณด์ํด์ ๋์จ๊ฒ ROUGE 2.0์ด๋ค.
* ROUGE 2.0์ synonymous(๋์์ด)์ topic coverage๋ฅผ ํฌํจํ์ฌ ์์ issue๋ฅผ ๋ณด์ํ์๋ค. โ `ROUGE - {NN | Topic | TopicUniq} + Synonyms`
* ๊ทธ๋ฌ๋ ์ฌ์ ํ ์๋ฒฝํ๊ฒ score ๋งค๊ธธ ์ ์์ง๋ง ํ์ฌ๊น์ง ๊ฐ์ฅ ์ข์ Evaluation ๋ฐฉ๋ฒ์ด๋ผ๊ณ  ํ๊ฐ๋ฐ๋๋ค.
<br>
<br>

## ๐ฌ ํฌ๊ณ  
* **workshop** 
  * ๋๊ท๋ชจ ํํ๋ ์์ํ  ๋ ์๋ค๋ก ํ๋ฃจ ๊ท๋ชจ์ workshop๋ฅผ ์งํํ๋ค. ๋ชฉ์ ์ ๋ณธ ํํ ์ฐธ์์๋ค์ด specificํ ํค์๋๋ฅผ ์ค์ฌ์ผ๋ก ๋ชจ์ฌ์ ์งํํ๋ ์์ ํํ๊ฐ์ ๋๋. ๋ณดํต ๋ณธ ํํ ๋ด๊ธฐ ์ ๋งคํ๊ฑฐ๋ Working in Process๋ฅผ ๊ณต์ ํ๊ณ  ํผ๋๋ฐฑ ๋ฐ๋ ์๋ฆฌ์ด๊ธฐ๋ ํ๋ค.
  * Call for workshop์ ์ด์ด committee๊ฐ pass/non pass ์ฌ๋ถ๋ฅผ ์ฃผ๊ณ  ๋ค์ ๊ทธ workshop์์ ๋ฐ์ ๋ผ๋ฌธ์ ๋ํ ๊ณต๊ณ ๋ฅผ ๋ธ๋ค. 
<br>

* **tutorial** 
  * ์๋ก์ด ๋ผ๋ฌธ์ ์ ์ํ๊ธฐ๋ณด๋ค๋ ๊ธ ๋ถ์ํ ์๋ก์ด ์ฃผ์ ์ ๋ํ ๊ฐ๋ก ์ ์ธ ๊ฐ์๋ฅผ ํ๋ ํ๋ฃจ ๊ท๋ชจ์ ์ธ์ (e.g. ACL 2020 open-domain QA tutorial)
<br>

* **full/short paper**
  * ํํ๋ง๋ค ๊ธฐ๋ํ๋ long/short paper์ ๋ํ ์คํ์ด ์๊ธฐ ๋๋ฌธ์, call for paper๋ฅผ ์ฐธ๊ณ ํ๋ ๊ฑธ ์ถ์ฒํ๋ค. ํต์์ ์ผ๋ก short paper๋ long paper์ ๋นํด ์๋นํ ์งง๊ณ  long paper ํฌ๊ณ ๊ฐ ๋ ์ธ์  ๋ฐ๋๋ค. 
  * NAACL call for papers 2022 
    * Long paper : (8 pages) substantial, original, completed and unpublished work
    * Short paper : (4 pages) original and unpublished work
<br>
<br>

## Dataset Download
|Dataset|Domain|Train|Val|Test|
|:---:|:---:|:---:|:---:|:---:|
|XSum|News|204,045|11,332|11,334|
|CNN/DM|News|287,113|13,368|11,490|

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

##  ๐  Study
### ๋๋๋น
* [๊ผผ๊ผผํ ๋ฅ๋ฌ๋ ๋ผ๋ฌธ ๋ฆฌ๋ทฐ์ ์ฝ๋ ์ค์ต](https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice)
* [์ปดํจํฐ ๊ณตํ๊ณผ๋ฅผ ์ํ ์ต์  ๋ผ๋ฌธ ์ฐพ์ ์ฝ๋ ๋ฐฉ๋ฒ ์ ๋ฆฌ](https://www.youtube.com/watch?v=FPcdxHCxH_o)
<br><br>

### Deep Learning 
* [PyTorch๋ก ์์ํ๋ ๋ฅ ๋ฌ๋ ์๋ฌธ](https://wikidocs.net/book/2788)
* [BERT ๋์๋ณด๊ธฐ](https://docs.likejazz.com/bert/)
<br><br>

### NLP basic
* [oh! suz's NLP Cookbook](https://www.ohsuz.dev/nlp-cookbook)
* [Jeonsworld's NLP ๊ด๋ จ ๋ผ๋ฌธ ๋ฆฌ๋ทฐ ํฌ์คํ](https://jeonsworld.github.io/)
* [์๊ฐ ์์ฐ์ด์ฒ๋ฆฌ - Facebook](https://www.facebook.com/monthly.nlp?hc_ref=ART_4x3Knm-Y_6Rw38lFMtWmKZ8SdL4fWSzm2I9CiaYwJAtFIHk9mP_T7mK69NC8V2A&fref=nf&__xts__[0]=68.ARD8SbISh91tv-3NTdye910Za6oW4Nkfc9S3jAAX3n9xWPQjLdTDJA9eCQh_J10Y3ROSXAR5k_zgzd7q77OEgRaN0yMMkp4XdSPzROUANUkOJajbcUBhbaPtD_riFOG2cAWkFIAJ35CE3XQvrYj4246-Ggebd06AhnUK_WuOr-nZFECcT_txc0ekAqJC_OEvZaGzcYr8CwWwjCCYO2cg3reKqV6CrF2unShmou5PdNlmFzpiNrmYlltICYZxFX-mQdn0eBXJkpxKBr_b_pD1LnBO2e0QcFI_cC6plzalWQ3RbB6daGM)
* [๋ฅ ๋ฌ๋์ ์ด์ฉํ ์์ฐ์ด ์ฒ๋ฆฌ ์๋ฌธ(Tensoflow/Keras ์ฌ์ฉ)](https://wikidocs.net/book/2155)
* [Generalized Language Models](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html#gpt)
