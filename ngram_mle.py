import numpy as np
from typing import Iterable, Sequence
from util import *

class ngramLMMLE(LM):
    def __init__(self, n : int, train_data : Iterable, debug: bool = False, unk: bool = False) -> None:
        self.train_data = train_data
        self.n = n
        self.N = len(train_data)

        if unk: #use the command line argument "--unk" to train with <unk> tokens
            self.train_unk()
        pass
        
        self.ngram_counts = get_counts(self.train_data, self.n)
        self.context_counts = get_counts(self.train_data, self.n - 1)

    def logprob(self, w : str, c : Sequence[str], debug : bool = False) -> float:
        ngram = " ".join(c + [w])
        context = " ".join(c)
        ngram_count = self.ngram_counts.get(ngram, 0)
        context_count = self.context_counts.get(context, 0)

        if context_count == 0:
            context_count = self.N

        if debug:
            print(ngram_count, context_count)
            print(ngram, "|", context)
            print(w, "|", context, "=", ngram_count/context_count)

        prob = ngram_count / context_count
        if prob > 0:
            return np.log(prob)
        else: 
            return float('-inf')
        
    def get_vocab(self) -> set[str]:
        vocab = set()
        vocab.update(self.train_data)
        return vocab
    
    def get_n(self) -> int:
        return self.n
    
    def train_unk(self) -> None:
        counts = get_counts(self.train_data, 1)
        for i in range(len(self.train_data)):
            if counts[self.train_data[i]] <= 1:
                self.train_data[i] = "<unk>"
        pass
            


