from wordle import create_wordle_response 
from collections import defaultdict
from math import log
from joblib import Memory
from multiprocessing import Pool
from functools import partial

memory = Memory('../artifacts', verbose=0)

def get_entropy(guess, solution_list):
    response_frequency = defaultdict(lambda: 0) # wordle response -> frequency 
    for hidden_word in solution_list:
        response = create_wordle_response(hidden_word=hidden_word, guess=guess)
        response_frequency[response] +=1
    count = float(len(solution_list))
    entropy = 0.0
    base = 2.0
    for response, frequency in response_frequency.items():
        probability = float(frequency)/count
        entropy += probability * log(probability, base)

    return guess, -1 * entropy

@memory.cache
def get_all_entropy(guess_list, solution_list, processes=1):
    results = [] # list of tuples of (word, entropy)
    if processes == 1: # good for unit tests
        for guess in guess_list:
            results.append(get_entropy(guess, solution_list))
    else:
        with Pool(processes=processes) as pool:
            partial_get_entropy = partial(get_entropy, solution_list=solution_list)
            for result in pool.imap_unordered(partial_get_entropy, guess_list):
                results.append(result)
    return results




