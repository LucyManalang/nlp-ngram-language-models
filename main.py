import pickle
import numpy as np
import argparse

from ngram_mle import ngramLMMLE
from ngram_laplace import ngramLMLaplace
from ngram_interp import ngramLMInterp
from ngram_backoff import ngramLMBackoff
from util import *

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True,
                        help="path to train data")
    parser.add_argument("--valid", type=str, required=True,
                        help="path to validation data to compute ppl for")
    parser.add_argument("--byword", action="store_true",
                        help="print logprobs for each word in the validation set")
    parser.add_argument("-n", type=int, required=True,
                        help="size of context window to consider")
    parser.add_argument("--smooth", type=str, default="mle",
                        help="smoothing technique (mle, laplace, interp, backoff)")
    parser.add_argument("--coeffs", type=str,
                        help="coeffs for interpolation, comma separated. provide n coefs, sum to 1.")
    parser.add_argument("--discount", type=float,
                        help="discount factor for backoff models. Between 0 and 1.")
    parser.add_argument("--load", type=str,
                        help="load a trained model from file")
    parser.add_argument("--save", type=str,
                        help="save a trained model to file")
    # Use this parameter to log what your algorithm does to help debugging!
    # The way you use this will not be graded, but any new print messages that
    # aren't conditioned on this flag being true WILL interfere with autograding
    parser.add_argument("--debug", action="store_true",
                        help="Print debug messages")
    parser.add_argument("--unk", action="store_true",
                        help="use <unk> tokens for unknown words")
    args = parser.parse_args()

    if args.load is None:
        # Load train data
        # data is lower-cased and split on whitespace.
        with open(args.train) as train_f:
            train_data = tokenize(train_f.read())

        # train the appropriate n-gram model
        # TODO: Modify the following lines to train a model that uses the smoothing technique 
        # indicated by args.smooth.
        if args.smooth == "mle":
            model = ngramLMMLE(args.n, train_data, debug=args.debug, unk=args.unk)
        elif args.smooth == "laplace":
            model = ngramLMLaplace(args.n, train_data, debug=args.debug, unk=args.unk)
        elif args.smooth == "interp":
            model = ngramLMInterp(args.n, train_data, debug=args.debug, unk=args.unk, coeffs=args.coeffs)
        elif args.smooth == "backoff":
            model = ngramLMBackoff(args.n, train_data, debug=args.debug, unk=args.unk, discount=args.discount)
        else:
            print("ERROR: Smoothing method {} specified in --smooth not recognized"
                  .format(args.smooth))
        
    else:
        with open(args.load, "rb") as load_f:
            model = pickle.load(load_f)

    # Save to file if requested
    if args.save is not None:
        with open(args.save, "wb") as save_f:
            pickle.dump(model, save_f)

    # Load validation data
    with open(args.valid) as valid_f:
        valid_data = tokenize(valid_f.read())

    print("ppl: {:.4}".format(get_ppl(model, valid_data, debug=args.debug, unk=args.unk)))
    # print("sum:", add_to_1(model, valid_data, debug=args.debug, unk=args.unk)) # use to calculate if model adds to 1

    if args.byword:
        for w, *c in ngram(valid_data, args.n):
            print("logp({}|{}) = {}".format(w, " ".join(c), model.logprob(w, c, debug=args.debug)))
