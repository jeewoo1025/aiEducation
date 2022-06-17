from transformers import BartForConditionalGeneration, BartTokenizer, PegasusTokenizer, PegasusForConditionalGeneration
import torch
import sys
import argparse
from typing import List

"""
    https://github.com/yixinL7/BRIO/blob/main/gen_candidate.py
"""

########################################################################################
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# log 출력 형식
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# log 출력
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# log를 파일에 출력
file_handler = logging.FileHandler('preprocessing.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
########################################################################################


def generate_summaries_cnndm(args):
    device = f"cuda:{args.gpuid}"
    mname = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(mname).to(device)
    model.eval()

    tok = BartTokenizer.from_pretrained(mname)
    max_length = 140
    min_length = 55
    count = 1
    bsz = 8         # batch size 

    with open(args.src_dir) as source, open(args.tgt_dir, 'w') as fout:
        sline = source.readline().strip().lower()
        slines = [sline]

        for sline in source:
            if count % 100 == 0:
                print(count, flush=True)

            if count % bsz == 0:
                with torch.no_grad():
                    dct = tok.batch_encode_plus(slines, max_length=1024, return_tensors="pt", pad_to_max_length=True, truncation=True)
                    summaries = model.generate(
                        input_ids=dct["input_ids"].to(device),
                        attention_mask=dct["attention_mask"].to(device),
                        num_return_sequences=16,
                        num_beam_groups=16, 
                        diversity_penalty=1.0,
                        num_beams=16,
                        max_length=max_length+2,     # +2 from original because we start at step=1 and stop before max_length
                        min_length=min_length+1,     # +1 from original because we start at step=1
                        no_repeat_ngram_size=3,
                        length_penalty=2.0,
                        early_stopping=True,
                    )
                    dec = [tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                
                for hypothesis in dec:
                    hypothesis = hypothesis.replace('\n', ' ')
                    fout.write(hypothesis + '\n')
                    fout.flush()
                slines = []
            
            sline = sline.strip().lower()
            if len(sline) == 0:
                sline = " "
            slines.append(sline)
            count += 1

        if slines != []:
            with torch.no_grad():
                dct = tok.batch_encode_plus(slines, max_length=1024, return_tensors="pt", pad_to_max_length=True, truncation=True)
                summaries = model.generate(
                    input_ids=dct["input_ids"].to(device),
                    attention_mask=dct["attention_mask"].to(device),
                    num_return_sequences=16, num_beam_groups=16, diversity_penalty=1.0, num_beams=16,
                    max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
                    min_length=min_length + 1,  # +1 from original because we start at step=1
                    no_repeat_ngram_size=3,
                    length_penalty=2.0,
                    early_stopping=True,
                )
                dec = [tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
            
            for hypothesis in dec:
                    hypothesis = hypothesis.replace("\n", " ")
                    fout.write(hypothesis + '\n')
                    fout.flush()


def generate_summaries_xsum(args):
    """
        Pegasus : https://huggingface.co/transformers/v4.1.1/model_doc/pegasus.html
    """
    device = f"cuda:{args.gpuid}"
    mname = "google/pegasus-xsum"
    model = PegasusForConditionalGeneration.from_pretrained(mname).to(device)
    model.eval()

    tok = PegasusTokenizer.from_pretrained(mname)
    count = 1
    bsz = 1     # bsz : batch size
    max_length = 64

    with open(args.src_dir) as source, open(args.tgt_dir, 'w') as fout:
        sline = source.readline().strip()
        slines = [sline]

        # logger.info(f"start enumerate source")

        for (i, sline) in enumerate(source):
            if count % 10 == 0:     # debug
                print(count)

            if count % bsz == 0:        # 2개씩
                # logger.info(f'count({count}) % bsz({bsz}) = 0')
                # logger.info(f"for문 i : {i} - sline : {sline}")
                
                with torch.no_grad():
                    """
                    ▶ tok.prepare_seq2seq_batch
                    https://huggingface.co/transformers/v4.1.1/main_classes/tokenizer.html#transformers.BatchEncoding
                    : prepare a batch that can be passed directly to an model

                    [ Params ]
                        src_texts = List of documents to summarize or source language texts
                        return_tensors : 'pt' (return PyTorch torch.Tensor objects)

                    [ Return ]
                        BatchEncoding (input_ids, attention_mask, labels)


                    ▶ generate
                    https://huggingface.co/docs/transformers/v4.19.2/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate
                    : diverse beam search 호출 by num_beams > 1 and num_beam_groups > 1

                    [ Params ]
                        **batch : input
                        num_return_sequences : the number of independently computed return sequences for each element in the batch 
                        num_beam_groups (G): number of groups to divide num_beams into in order to ensure diversity among different groups of beams
                        num_beams (B): number of beams for beam search (1 means no beam search). (num beam groups <= num_beams)
                        diversity_penalty : 다른 group의 beam에서 동일한 token을 생성하는 경우 beam score에서 차감됨
                        length_penalty : exponential penalty to the length

                    [ Return ]
                        torch.LongTensor : [num_return_sequences x dim]


                    ▶ batch_decode
                    : convert a list of lists of token ids into a list of strings by calling decode

                    [ Params ]
                        sequences : torch.Tensor, List of tokenized input ids
                        skip_special_tokens : whether or not to remove special tokens in the decoding (default : False)

                    [ Return ]
                        List[str] : the list of decoded sentences
                    """
                    batch = tok.prepare_seq2seq_batch(src_texts=slines, return_tensors="pt").to(device)
                    gen = model.generate(**batch, num_return_sequences=128, num_beam_groups=16, diversity_penalty=0.1, num_beams=128, length_penalty=0.6)
                    dec: List[str] = tok.batch_decode(gen, skip_special_tokens=True)
                
                dec = [dec[i] for i in range(len(dec)) if i%8 == 0]     # 128(num beams) % 16 (num_beam_groups) = 8 (B'), 각 G의 첫번째 문장만 candi로 dec
                # logger.info(f'dec : {dec} \n\n')
                for hypothesis in dec:
                    fout.write(hypothesis + '\n')
                    fout.flush()
                slines = []

            sline = sline.strip()
            if len(sline) == 0:
                sline = " "
            slines.append(sline)
            count += 1
        
        if slines != []:
            # logger.info(f"slines != [] : {slines}")
            with torch.no_grad():
                batch = tok.prepare_seq2seq_batch(src_texts=slines, return_tensors="pt").to(device)
                gen = model.generate(**batch, num_return_sequences=128, num_beam_groups=16, diversity_penalty=0.1, num_beams=128, length_penalty=0.6)
                dec: List[str] = tok.batch_decode(gen, skip_special_tokens=True)
                
            dec = [dec[i] for i in range(len(dec)) if i%8 == 0]     # 128(num beams) % 16 (num_beam_groups) = 8 (B'), 각 G의 첫번째 문장만 candi로 dec
            for hypothesis in dec:
                fout.write(hypothesis + '\n')
                fout.flush()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--gpuid", type=int, default=0, help="gpu id")
    parser.add_argument("--src_dir", type=str, help="source file")
    parser.add_argument("--tgt_dir", type=str, help="target file")
    parser.add_argument("--dataset", type=str, default="xsum", help="dataset")
    args = parser.parse_args()

    if args.dataset == "xsum":
        generate_summaries_xsum(args)
