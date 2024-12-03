import numpy as np
from typing import Iterable, Sequence, Protocol, Mapping


class LM(Protocol):
    def logprob(self, w : str, c : Sequence[str], debug: bool, unk: bool) -> float: ...
    def get_n(self) -> int: ...
    def get_vocab(self) -> Sequence[str]: ...

def tokenize(data: str) -> list[str]:
    return data.lower().split()

def get_ppl(model: LM, eval_data: Iterable[str], debug: bool = False, unk: bool = False) -> float:
    eval_data = pad(eval_data, model.get_n())
    if unk:
        eval_data = replace_unk(eval_data, model.get_vocab())
        
    n = model.get_n()
    N = len(eval_data)
    log_prob_sum = 0.0
    for i in range(n-1, N):
        context = eval_data[i - n + 1:i]
        word = eval_data[i]

        logprob = model.logprob(word, context, debug)
        log_prob_sum += logprob
        
        if debug:
            print("logp({}|{}) = {:.3}".format(word, " ".join(context), logprob))
    return np.exp(-log_prob_sum / N) 

# use to calculate if model adds to 1. Set logprob return to prob and eval_data to brown.vocab.txt
def add_to_1(model: LM, eval_data: Iterable[str], debug: bool = False, unk: bool = False) -> float:
    num = 0
    for i in eval_data:
        num += model.logprob(i, [], debug)
    return num / len(eval_data)

def ngram(tokens: Iterable[str], n: int) -> Iterable[Sequence[str]]:
    ngram_seq = []
    for i in range(len(tokens) - n + 1):
        ngram_seq.append(tuple(tokens[i:i+n]))
    return ngram_seq

def pad(words : Sequence[str], n : int) -> Sequence[str]:
    new_words = []
    for i in range(n - 1):
        new_words.append("<eos>")
    for i in range(len(words)):
        new_words.append(words[i])
        if words[i] == "<eos>":
            for j in range(n - 1):
                new_words.append("<eos>")
    return new_words

def get_counts(train : Sequence[str], n : int) -> Mapping[str, int]:
    counts = {}
    train = pad(train, n)
    train = ngram(train, n)
    
    for i in train:
        j = " ".join(i)
        if j in counts:
            counts[j] += 1
        else:
            counts[j] = 1
    return counts

def replace_unk(data : Sequence[str], vocab : Sequence[str]) -> Sequence[str]:
    new_data = []
    for i in data:
        if i not in vocab:
            new_data.append("<unk>")
        else:
            new_data.append(i)
    return new_data