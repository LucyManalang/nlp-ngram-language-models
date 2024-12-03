import numpy as np
from typing import Iterable, Sequence
from util import *

class ngramLMInterp(LM):
    def __init__(self, n : int, train_data : Iterable, debug: bool = False, unk: bool = False, coeffs: str = None) -> None:
        self.train_data = train_data
        self.counts = []
        self.lambdas = []
        self.n = n
        self.N = len(train_data)

        if unk: #use the command line argument "--unk" to train with <unk> tokens
            self.train_unk()
        if coeffs is None:
            self.lambdas = [1.0/n] * n
        else:
            for i in coeffs.split(","):
                self.lambdas.append(float(i))
        
        self.vocab = self.get_vocab()

        for i in range(1, n + 1):
            self.counts.append(get_counts(self.train_data, i))
        pass

    def logprob(self, w : str, c : Sequence[str], debug : bool = False) -> float:
        prob = 0.0
        for i in range(self.n - 1, -1, -1):

            ngram = " ".join(c[-i:] + [w]) if i > 0 else w
            context = " ".join(c[-i:]) if i > 0 else ""

            ngram_count = self.counts[i].get(ngram, 0)
            context_count = self.counts[i].get(context, 0)

            if context_count == 0:
                context_count = self.N

            prob += self.lambdas[i] * (ngram_count / context_count)

            if debug:
                print(w, "|", context, "=", prob)
                print(ngram_count, context_count)
        if prob > 0:
            return prob
        else:
            return -np.inf
        
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
        

    