[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/nqYH_z4M)
# HW1: N-Gram Language Models

In Homework 1, we'll be training and analyzing the performance of n-gram language models. This assignment will build directly off of the n-gram activity, where you fit the parameters of an n-gram language model using Maximum Likelihood Estimation (MLE) --- that is, building an n-gram model that maximises the likelihood of the training data.

In this assignment, you'll be asked to 

1. Implement and train N-Gram models that implement various kinds of *smoothing* (laplace smoothing, interpolation, and simple backoff).

2. Quantitatively compare n-gram models with various choices of n, smoothing techniques, and data using perplexity.

3. Qualitatively compare the outputs of selected models by sampling from the LM's distribution.

### Submission 
You must include in your submission

1. Working code implementing the requirements outlined in Part 1 using the provided command-line interface.
2. A report named `HW1.pdf` in the top directory of your submission (with content requirements outlined in parts 2 and 3).
3. Code that can reproduce results provided in the report.

Submit by pushing to this repository (as created by Github classroom). 

It is recommended (but not strictly required) that you submit your report using the [style required for the Association for Computational Linguistics Conferences](https://github.com/acl-org/acl-style-files). There are templates for both Latex (recommended!) and MS Word (not recommended, but allowed). 

## Part 0: Starter Code

Familiarize yourself with the starter code. 

- `main.py`        provides you with code that implements a standardized command-line interface for your code, as well as model saving/loading functionality through pickle.

- `ngram_mle.py`    provides a template to start building a class to encapsulate an ngram LM's functionality. 

- `util.py`         provides a location to store utility code (i.e., code that is shared across multiple models, perhaps stored in separate files). 

TODOs have been placed where provided code will need to be modified as you complete the assignment, but DO NOT modify command-line parameters: this will cause you to lose points to auto-grading and will make grading the rest of your assignment more difficult.

Also be sure to check out the `data/` directory!

- `brown.{train/valid}.txt` contains the training/validation set I'd like you to focus on for the analysis below (though feel free to supplement with other files!). This is taken from the [Brown corpus](http://gandalf.aksis.uib.no/icame/brown/bcm.html), a corpus of American English drawn from various sources constructed by Francis & Kucera as part of a collaboration between Brown University and the US Department of Education. This is a fairly old, but standard corpus in NLP --- so standard that I constructed the train/validation split using NLTK!
- `brown.vocab.txt` contains a newline-separated list of every token that appears at least once in the brown corpus. This should be useful for constructing the vocabulary of your model (i.e., consider how you might implement Laplace smoothing for words you don't encounter until the validation set, since we don't use `<unk>` tokens in this homework).
- `austen-*.txt` Two texts (Emma, Sense and Sensibility) written by Jane Austen taken from the Gutenberg Corpus (also sourced and compiled through NLTK!). May be useful in case you want to consider training models on text from a different style.

## Part 1: Implementing n-grams with smoothing

**Write code that will train n-gram models with Laplace (i.e., add-1) smoothing, interpolation (with user-specified interpolation weights), and simple backoff.**

For clarity, 
- Laplace smoothing should operate like the MLE n-gram models you trained during class, but add 1 to every count (including unseen n-gram combinations). 
- Interpolation should, given a set of weights $\lambda_1 ... \lambda_n$ such that $\sum_{1 \leq i \leq n} \lambda_i = 1$, return a weighted average of models from an MLE unigram to an MLE n-gram where an MLE k-gram has weight $\lambda_k$. That is, if $P_k(w \mid c)$ is the conditional probability of the word $w$ appearing after the context $c$ under an MLE k-gram model, an interpolation model should estimate P(w \mid c) =  $\sum_{1 \leq k \leq n} \lambda_kP_k(w \mid c)$.  
- Simple backoff should, given a discounting factor $\delta$, estimate $P(w \mid c) = \delta P_n(w \mid c)$ (where, again, P_n is the distribution from an MLE estimated n-gram model) *if* $P_n(w \mid c) > 0$ (i.e., we saw the n-gram c + w!). Otherwise, we back off to the (n-1)-gram model, and estimate $P(w \mid c) = \delta(1-\delta) P_{n-1}(w \mid c)$ if $P_{n-1}(w \mid c) > 0$. If not, we repeat, discounting the lower-order n-gram model until we reach the unigram model, where, if we reach it, we assign $(1-\delta)^{n-1}P_1(w)$. 

I recommend that you make sure you understand the math behind each smoothing technique before attempting to implement any of them. If you can't determine the probabilitities by hand, you'll have trouble testing your implementation!

## Part 2: Quantitative Analysis

**Implement a function to compute the perplexity of a model over a dataset and compare the complexities of trained n-gram models to answer the following questions**

For each of the following questions, train and evaluate a set of models to attempt to answer the question. Use the results you collect to argue for your answer in the write-up. 

1. How do different implementations of smoothing (or lack of smoothing) affect *training set* perplexity?

2. How do different implementations of smoothing (or lack of smoothing) affect *validation set* perplexity?

3. How does your choice of $n$ affect the perplexity of your model? Does adding smoothing affect your answer?

Note that there is no exact "correct" set of analyses I want you to provide. Moreso, I'm looking for your thought process designing tests to answer these questions, your awareness of potential limitations in your approach (you certainly won't be trying all possible choices of $n$!), and your ability to use limited empirical evaluations to craft an argument about your choice of hyperparameters.

## Part 3: Qualitative Analysis

**Write code that allows you to sample from the n-gram language models you've trained, and qualitatively evaluate the output**. Consider looking at the Probability and Simulations activity to see techniques to sample from a finite, discrete probability distribution (as your n-gram model provides). 

Write at least 5 *preambles* (partial sentences) in a file called preambles.txt. Design these sentences to aid in your analysis of the models (Consider your choice of training data, etc.). Ask yourself the following questions:

1. What would you (or a typical speaker of English) expect to complete the sentence?
2. What would you expect a model trained in the manner an n-gram LM is to complete this sentence with? Is this different than 1? Do you expect the model to make a mistake (however you define that)?

Generate 10 completions for each of the preambles written (load them from preambles.txt) for a variety of values of n and smoothing techniques (at least 5 models). You should select models that you've trained in Part 2, and should select them so you can build intuitions about now your choice of n and smoothing techniques affects how natural your n-gram LM generations are. **For each, record your answers to the above questions in your report, as well as an assessment of how the model performed given your design. Were you surprised?**. Then, conclude this section by summarizing your findings across the models you've analyzed and try to draw what conclusions you have.   

## Advice for this assignment

1. You should write your own suite of tests (consider using [unittest](https://docs.python.org/3/library/unittest.html) or a 3rd-party alternative). Proper testing likely involves manually counting n-gram occurances for a small training "corpus" and determining the adjusted probabilities each smoothing method would estimate. 

2. The report is not an afterthought! One of our learning goals is to be able to choose appropriate experiments to run to compare different NLP systems --- the analyses and the report allows us to do that with respect to a hyperparameter (choice of n, or choice of coeffs/discount factor) and architecture (smoothing techniques). One implicit goal is to have you practice *communicating* the experiments you chose and their results. These will be practice for your final assignment!

3. The starter code contains type hints, as specified in the [python docs](https://docs.python.org/3/library/typing.html). This is mostly here to get you some passive exposure to how some of the assurances of static typing (like you're used to in Java/C) can be used in dynamically typed languages like Python. Note that, by themselves, type annotations are only a guide for programmers --- the standard python runtime doesn't check these! You can, however, use 3rd party tools (mypy and pyright are the major ones). If your code doesn't like the annotations, make sure you're running it in a Python 3.10+ environment! 

