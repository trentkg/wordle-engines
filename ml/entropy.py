from wordle import create_wordle_response 
from collections import defaultdict
from math import log
from functools import cache

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

    return -1 * entropy

@cache
def get_all_entropy(guess_list, solution_list):
    results = [] # list of tuples of (word, entropy)
    for guess in guess_list:
        entropy = get_entropy(guess, solution_list)
        results.append((guess, entropy))
    return results
