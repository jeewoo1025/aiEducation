# All about Coding
`written by 설지우` <br>
__Last Updated 2025-06-15__

## 📐 Metrics

### 1. **Pass@k** (Functional Correctness)
k개의 샘플 중 최소 하나가 정답이면 성공으로 간주하는 가장 널리 사용되는 메트릭 [[link](https://deepgram.com/learn/humaneval-llm-benchmark?utm_source=chatgpt.com)]

```python
# https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py
def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])
```

### 2. **Resolved Rate** (Issue Resolution)
실제 소프트웨어 이슈를 성공적으로 해결한 비율을 측정하는 메트릭 (SWE-bench에서 사용)

```python
# SWE-bench evaluation process
def calculate_resolved_rate(instances_resolved: int, total_instances: int) -> float:
    """
    Calculate the percentage of instances successfully resolved.
    
    Args:
        instances_resolved: Number of instances where patch fixed the issue
        total_instances: Total number of instances in the dataset
    
    Returns:
        Resolved rate as percentage
    """
    return (instances_resolved / total_instances) * 100.0

# Evaluation steps:
# 1. Set up Docker environment for repository
# 2. Apply model's generated patch
# 3. Run repository's test suite
# 4. Determine if patch successfully resolves the issue
```

### 3. **BLEU** (Syntactic Similarity)
생성된 코드가 정답 코드와 토큰 수준에서 얼마나 유사한지 평가

```python
# https://github.com/hendrycks/apps/blob/main/eval/eval_bleu.py

import sacrebleu
from sacremoses import MosesDetokenizer
md = MosesDetokenizer(lang='en')

random.seed(12345678987654321)

def calc_bleu(output: List[str], targets: List[List[str]]):
    """
    Calculate BLEU score for generated code against reference solutions.
    """
    max_bleu = 0
    bleu = sacrebleu.corpus_bleu(output, targets)
    for item in targets[0]:
        tmp_bleu = sacrebleu.corpus_bleu(output, [[item]])
        if tmp_bleu.score > max_bleu:
            max_bleu = tmp_bleu.score
    return bleu.score, max_bleu
```

### 4. **CodeBLEU** (Syntactic + Semantic Similarity)
BLEU의 확장으로 AST(Abstract Syntax Tree)와 데이터 플로우를 고려한 코드 특화 메트릭

```python
# CodeBLEU combines multiple components:
# - BLEU: n-gram overlap
# - Weighted n-gram match: syntax-aware token matching
# - AST match: syntactic structure similarity  
# - Data-flow match: semantic variable relationships

def calculate_codebleu(
    predictions: List[str], 
    references: List[List[str]], 
    lang: str,
    weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
) -> float:
    """
    Calculate CodeBLEU score.
    
    Args:
        predictions: Generated code snippets
        references: Reference code solutions
        lang: Programming language
        weights: Weights for (bleu, weighted_ngram, ast, dataflow)
    
    Returns:
        CodeBLEU score (0-100)
    """
    # Implementation involves AST parsing and dataflow analysis
    pass
```

### 5. **Execution Accuracy** (Binary Correctness)
생성된 코드가 주어진 테스트 케이스를 모두 통과하는지 이진 평가

```python
def execution_accuracy(generated_code: str, test_cases: List[dict]) -> bool:
    """
    Evaluate if generated code passes all test cases.
    
    Args:
        generated_code: The generated code snippet
        test_cases: List of input-output test cases
    
    Returns:
        True if all test cases pass, False otherwise
    """
    try:
        exec(generated_code, globals())
        for test_case in test_cases:
            result = eval(f"solution({test_case['input']})")
            if result != test_case['expected_output']:
                return False
        return True
    except Exception:
        return False
```

### 6. **Edit Similarity** (Code Modification)
원본 코드 대비 수정된 부분의 정확성을 평가 (code editing 작업용)

```python
import difflib

def edit_similarity(original: str, generated: str, reference: str) -> float:
    """
    Calculate edit similarity based on diff operations.
    
    Args:
        original: Original buggy code
        generated: Model-generated fixed code  
        reference: Reference solution
    
    Returns:
        Similarity score (0-1)
    """
    gen_diff = list(difflib.unified_diff(original.split('\n'), generated.split('\n')))
    ref_diff = list(difflib.unified_diff(original.split('\n'), reference.split('\n')))
    
    # Compare edit operations
    common_edits = set(gen_diff) & set(ref_diff)
    total_edits = len(set(gen_diff) | set(ref_diff))
    
    return len(common_edits) / total_edits if total_edits > 0 else 0.0
```

## 메트릭 비교 및 선택 가이드

| 메트릭 | 평가 대상 | 장점 | 단점 | 적용 벤치마크 |
|--------|-----------|------|------|---------------|
| **Pass@k** | 기능적 정확성 | • 실제 실행 결과 기반<br>• 다양한 솔루션 허용 | • 테스트 케이스 품질 의존<br>• 계산 비용 높음 | HumanEval, MBPP, LiveCodeBench |
| **Resolved Rate** | 실무 문제 해결 | • 현실적 평가<br>• 전체 워크플로우 고려 | • 환경 설정 복잡<br>• 재현성 이슈 | SWE-bench, SWE-bench Verified |
| **BLEU** | 구문적 유사성 | • 빠른 계산<br>• 자연어 처리에서 검증됨 | • 코드 특성 미반영<br>• 정확한 코드도 낮은 점수 | APPS (보조 메트릭) |
| **CodeBLEU** | 구문+의미적 유사성 | • 코드 구조 고려<br>• AST와 데이터플로우 반영 | • 언어별 파서 필요<br>• 복잡한 계산 | CodeXGLUE, 코드 번역 작업 |
| **Execution Accuracy** | 이진 정확성 | • 명확한 기준<br>• 빠른 평가 | • 부분 점수 없음<br>• 엄격한 평가 | 초기 코드 생성 연구 |
| **Edit Similarity** | 수정 정확성 | • 수정 작업에 특화<br>• 변경량 고려 | • 구현 복잡<br>• 기준 모호 | 코드 수정, 버그 픽스 |

## 권장 사용법

1. **기본 평가**: Pass@1, Pass@10을 주 메트릭으로 사용
2. **보조 평가**: CodeBLEU로 구조적 유사성 측정  
3. **실무 평가**: SWE-bench의 Resolved Rate로 실제 적용성 검증
4. **다중 메트릭**: 여러 관점에서 종합적 평가 권장

<br>

## 📊 Benchmark Overview

| 벤치마크 이름 | 지원 언어 | 언어 수 | 문제 수 | 주요 메트릭 | 설명 | 링크 |
|-------------|----------|---------|---------|------------|------|------|
| **HumanEval** | Python | 1개 | 164개 | Pass@1, Pass@10, Pass@100 | OpenAI에서 개발한 함수 수준의 코드 생성 벤치마크. 가장 널리 사용되는 기준점 | [GitHub](https://github.com/openai/human-eval) |
| **HumanEval+** | Python | 1개 | 164개 | Pass@1, Pass@10, Pass@100 | HumanEval의 개선 버전으로 더 엄격한 테스트 케이스 포함 | [GitHub](https://github.com/evalplus/evalplus) |
| **HumanEval-X** | Python, C++, Java, JavaScript, Go | 5개 | 164개 × 5개 언어 | Pass@1, Pass@10, Pass@100 | HumanEval을 다중 언어로 확장한 벤치마크 | [GitHub](https://github.com/THUDM/CodeGeeX) |
| **MBPP** | Python | 1개 | 974개 | Pass@1, Pass@10, Pass@100 | Mostly Basic Programming Problems. 초급~중급 수준 프로그래밍 문제 | [GitHub](https://github.com/google-research/google-research/tree/master/mbpp) |
| **MBPP+** | Python | 1개 | 974개 | Pass@1, Pass@10, Pass@100 | MBPP의 개선 버전으로 더 엄격한 테스트 케이스 포함 | [GitHub](https://github.com/evalplus/evalplus) |
| **CodeContests** | C++, Python, Java | 3개 | 13,610개 | Pass@1, Pass@10, Pass@100 | DeepMind의 경쟁 프로그래밍 문제 데이터셋 | [GitHub](https://github.com/deepmind/code_contests) |
| **MultiPL-E** | Python, C++, Java, JavaScript, TypeScript, PHP, C#, Bash, R, Perl, Scala, Swift, Rust, Go, D, Julia, Lua, Racket | 18개 | 164개 × 18개 언어 | Pass@1, Pass@10, Pass@100 | HumanEval을 18개 언어로 확장한 대규모 다국어 벤치마크 | [GitHub](https://github.com/nuprl/MultiPL-E) |
| **LiveCodeBench** | Python, C++, Java, JavaScript | 4개 | 300+개 (지속 증가) | Pass@1 | 실시간으로 업데이트되는 오염 방지 벤치마크. LeetCode, AtCoder, CodeForces에서 수집 | [Website](https://livecodebench.github.io/) |
| **BigCodeBench** | Python | 1개 | 1,140개 | Pass@1, Pass@10, Pass@100 | HumanEval의 차세대 벤치마크로 더 복잡하고 실용적인 코딩 작업 포함 | [GitHub](https://github.com/bigcode-project/bigcodebench) |
| **DS-1000** | Python | 1개 | 1,000개 | Pass@1, Pass@10, Pass@100 | 데이터 사이언스 작업에 특화된 벤치마크 (NumPy, Pandas, SciPy, Matplotlib 등) | [GitHub](https://github.com/HKUNLP/DS-1000) |
| **APPS** | Python | 1개 | 10,000개 | Pass@1, Pass@5, Pass@100 | 경쟁 프로그래밍과 면접 문제를 포함한 대규모 벤치마크 | [GitHub](https://github.com/hendrycks/apps) |
| **SWE-bench** | Python | 1개 | 2,294개 | Pass@1, Resolved Rate | 실제 GitHub 저장소의 이슈 해결 능력을 평가하는 소프트웨어 엔지니어링 벤치마크 | [GitHub](https://github.com/princeton-nlp/SWE-bench) |

## 주요 특징

### 언어 지원 현황
- **단일 언어 (Python만)**: HumanEval, MBPP, BigCodeBench, DS-1000, APPS, SWE-bench
- **소수 언어 (2-5개)**: HumanEval-X (5개), CodeContests (3개), LiveCodeBench (4개)
- **다국어 (10개 이상)**: MultiPL-E (18개)

### 메트릭 설명
- **Pass@1**: 1번 시도에서 정답을 생성할 확률
- **Pass@10**: 10번 시도 중 적어도 1번 정답을 생성할 확률  
- **Pass@100**: 100번 시도 중 적어도 1번 정답을 생성할 확률
- **Resolved Rate**: 실제 소프트웨어 이슈 해결 성공률

### 벤치마크 트렌드
1. **포화 현상**: HumanEval Pass@1이 99.4%, MBPP가 94.2%에 도달하여 새로운 벤치마크 필요성 증대
2. **오염 방지**: LiveCodeBench는 시간별 문제 수집으로 데이터 오염 방지 (2025.06 기준 `release_v6` 공개)
3. **실용성 강화**: BigCodeBench, SWE-bench 등 실제 소프트웨어 개발 시나리오 반영
4. **다국어 확장**: MultiPL-E는 18개 언어 지원으로 가장 포괄적

<br>

## Github Example Code
### BigCode Evaluation Harness 
- GitHub: https://github.com/bigcode-project/bigcode-evaluation-harness
- 주요 특징:
    - EleutherAI/lm-evaluation-harness에서 영감을 받은 코드 생성 모델 평가 프레임워크
    - 배치 추론 지원 (--batch_size 옵션)
    - HumanEval, MBPP, MultiPL-E, DS-1000 등 다양한 벤치마크 지원
    - 멀티 GPU 지원 (accelerate launch)

```bash
accelerate launch main.py \
    --model codellama/CodeLlama-7b-Python-hf \
    --tasks humaneval \
    --temperature 0.1 \
    --n_samples 10 \
    --batch_size 8 \
    --allow_code_execution
```

### Basic Code: Eval StarCoder
- [abacaj/code-eval](https://github.com/abacaj/code-eval/tree/main)
```python
from transformers import (
    AutoTokenizer,
    GPTBigCodeForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from core import run_eval, filter_code, fix_indents
import os
import torch

# TODO: move to python-dotenv
# add hugging face access token here
TOKEN = ""


@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str, batch_size: int
) -> list[str]:
    input_batch = [prompt for _ in range(batch_size)]
    inputs = tokenizer(input_batch, return_tensors="pt").to(model.device)
    input_ids_cutoff = inputs.input_ids.size(dim=1)

    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,  # model has no pad token
    )

    batch_completions = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids],
        skip_special_tokens=True,
    )

    # fix_indents is required to fix the tab character that is generated from starcoder model
    return [filter_code(fix_indents(completion)) for completion in batch_completions]


if __name__ == "__main__":
    # adjust for n = 10 etc
    num_samples_per_task = 10
    out_path = "results/starcoder/eval.jsonl"
    os.makedirs("results/starcoder", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        "bigcode/starcoder",
        trust_remote_code=True,
        use_auth_token=TOKEN,
    )

    model = torch.compile(
        GPTBigCodeForCausalLM.from_pretrained(
            "bigcode/starcoder",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            max_memory={
                0: "18GiB",
                1: "18GiB",
            },
            use_auth_token=TOKEN,
        ).eval()
    )

    run_eval(
        model,
        tokenizer,
        num_samples_per_task,
        out_path,
        generate_batch_completion,
        True,
    )
```

<br>

## 📚 References

- **HumanEval**  
  Chen et al., *Evaluating Large Language Models Trained on Code*, arXiv 2021.  
  [[Paper](https://arxiv.org/abs/2107.03374)] • [[GitHub](https://github.com/openai/human-eval)]

- **MBPP (Mostly Basic Programming Problems)**  
  Austin et al., *Program Synthesis with Large Language Models*, arXiv 2021.  
  [[Paper](https://arxiv.org/abs/2108.07732)] • [[GitHub](https://github.com/google-research/google-research/tree/master/mbpp)]

- **APPS**  
  Hendrycks et al., *Measuring Coding Challenge Competence With APPS*, NeurIPS 2021.  
  [[Paper](https://arxiv.org/abs/2105.09938)] • [[GitHub](https://github.com/hendrycks/apps)]

- **CodeContests**  
  Nijkamp et al., *CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis*, arXiv 2022.  
  [[Paper](https://arxiv.org/abs/2203.13474)] • [[GitHub](https://github.com/salesforce/CodeGen)]

- **DS-1000**  
  Hong et al., *A Data Science Benchmark for Code Generation: DS-1000*, EMNLP 2023.  
  [[Paper](https://arxiv.org/abs/2305.14764)] • [[GitHub](https://github.com/HKUNLP/DS-1000)]

- **SWE-bench**  
  Khatami et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*, arXiv 2023.  
  [[Paper](https://arxiv.org/abs/2310.06770)] • [[Website](https://www.swebench.com/)] • [[GitHub](https://github.com/princeton-nlp/SWE-bench)]

- **SWE-bench Lite / Multimodal / Verified**  
  SWE-bench 팀 공식 웹사이트 기반.  
  [[Lite](https://www.swebench.com/lite.html)] • [[Multimodal](https://www.swebench.com/multimodal.html)]

- **LeetCode**  
  https://leetcode.com (문제 및 테스트 기반 벤치마킹, 공식 논문 없음)

---
