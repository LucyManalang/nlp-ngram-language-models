import pickle
import numpy as np
from scipy import stats
import random

from ngram_mle import ngramLMMLE
from ngram_laplace import ngramLMLaplace
from ngram_interp import ngramLMInterp
from ngram_backoff import ngramLMBackoff
from util import *

class LM(Protocol):
    def logprob(self, w : str, c : Sequence[str], debug: bool, unk: bool) -> float: ...
    def get_n(self) -> int: ...
    def get_vocab(self) -> Sequence[str]: ...

def get_preambles() -> Iterable[str]:
    with open("preambles.txt") as f:
        return list(f.read().splitlines())
    
def get_train_data(filename: str) -> Iterable[str]:
    with open(filename) as f:
        return f.read().split(" ")

def complete_sentence(model: LM, preamble: str, max_length: int = 20, unk: bool = False, debug: bool = False) -> str:
    generated = preamble[:]
    vocab = {}
    vocab = list(model.get_vocab())
    
    for _ in range(max_length):
        log_probs = []
        for word in vocab:
            log_prob = model.logprob(word, preamble[-(model.get_n() - 1):], debug=False)
            log_probs.append(log_prob)
        
        probs = []
        for i in range(len(log_probs)):
            probs.append(np.exp(log_probs[i]))
            probs[i] /= 50442

        map = {vocab[i]: probs[i] for i in range(len(vocab))}
        if debug:
            log_probs_dict = {word: probs for word, probs in zip(vocab, probs)}
            sorted_probs = dict(sorted(log_probs_dict.items(), key=lambda item: item[1], reverse=True))
            sorted_probs_keys = list(sorted_probs.keys())
            sorted_probs_values = list(sorted_probs.values())
            print(f"Most probable word: {sorted_probs_keys[0]} with probability: {sorted_probs_values[0]}")
            print(f"Least probable word: {sorted_probs_keys[-1]} with probability: {sorted_probs_values[-1]}")
        word = list(map.keys())
        next_word = random.choices(word, probs)[0]
        
        if next_word == "<eos>":
            break

        next_word = next_word.lower()
        generated.append(next_word)
        preamble.append(next_word)
    return " ".join(generated)

if __name__ == "__main__":
    # model = ngramLMMLE(n=2, train_data=get_train_data("./data/brown.train.txt"), debug=False, unk=False)
    # model = ngramLMLaplace(n=5, train_data=get_train_data("./data/brown.train.txt"), debug=False, unk=False)
    # model = ngramLMInterp(n=5, train_data=get_train_data("./data/brown.train.txt"), debug=False, unk=False, coeffs=".1,.1,.1,.2,.5")
    model = ngramLMBackoff(n=5, train_data=get_train_data("./data/brown.train.txt"), debug=False, unk=False, discount=0.4)
    

    for i in get_preambles():
        preamble = pad(tokenize(i), model.get_n())
        completed_sentence = complete_sentence(model, preamble, max_length=20, unk=False)
        completed_sentence = completed_sentence.replace("<eos> ", "") #uncomment to remove <eos> from the sentence
        print("Preamble: ", i)
        print("Generated sentence: ", completed_sentence)
