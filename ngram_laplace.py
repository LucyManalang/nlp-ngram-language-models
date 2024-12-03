import numpy as np
from typing import Iterable, Sequence
from util import *

class ngramLMLaplace(LM):
    def __init__(self, n : int, train_data : Iterable, debug: bool = False, unk: bool = False) -> None:
        self.train_data = train_data

        if unk: #use the command line argument "--unk" to train with <unk> tokens
            self.train_unk()

        self.ngram_counts = get_counts(self.train_data, n)
        self.context_counts = get_counts(self.train_data, n - 1)
        self.n = n
        self.N = len(self.train_data)
        pass

    def logprob(self, w : str, c : Sequence[str], debug : bool = False) -> float:
        ngram = " ".join(c + [w])
        context = " ".join(c)

        ngram_count = self.ngram_counts.get(ngram, 0) + 1
        context_count = self.context_counts.get(context, 0) + self.N
        prob = ngram_count / context_count
        return np.log(prob)
    
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


    


