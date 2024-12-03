import numpy as np
from typing import Iterable, Sequence
from util import *

class ngramLMBackoff(LM):
    def __init__(self, n : int, train_data : Iterable, discount : float, debug: bool = False, unk: bool = False) -> None:
        self.train_data = train_data
        self.counts = []
        self.n = n

        if unk:
            self.train_unk()

        self.N = len(self.train_data)
        self.discount = discount
       
        self.vocab = self.get_vocab()

        for i in range(1, n + 1):
            self.counts.append(get_counts(self.train_data, i))
        pass


    def recursive_backoff(self, w : str, c: Sequence[str], order: int, backoff_count : int = 1, debug : bool = False) -> float:
        if order == 1:
            ngram = w
            context = ""
        else:
            ngram = " ".join(c[-(order-1):] + [w])
            context = " ".join(c[-(order-1):])

        ngram_count = self.counts[order-1].get(ngram, 0)
        context_count = self.counts[order-2].get(context, 0) if order > 1 else self.N
        
        if debug: 
            print(f"Order: {order}, Ngram: '{ngram}', Context: '{context}'")
            print(f"Ngram count: {ngram_count}, Context count: {context_count}")

        if context_count > 0:
            prob = ngram_count / context_count
            if debug:
                print(f"{ngram} probability: {prob}")
                print(f"{order} ngram {ngram_count}")
                print(f"{order} context {context_count}")
            if prob > 0 or order == 1: 
                if debug:
                    print(f"{ngram} should end here probability: {prob} and log prob {np.log(prob)}")
                return np.log(prob) # don't need to multiply by discount factor since probability will always be zero if we backoff 
            else:
                prob = self.recursive_backoff(w, c, order - 1, backoff_count + 1, debug) * ((1 - self.discount) ** backoff_count)
                return prob # it's already been np.logged in the recursive function
        if context_count == 0:
            prob = self.recursive_backoff(w, c, order - 1, backoff_count + 1, debug) * ((1 - self.discount) ** backoff_count)
            return prob

    def logprob(self, w : str, c : Sequence[str], debug : bool = False) -> float:
        return self.recursive_backoff(w, c, self.n, debug)
    
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
