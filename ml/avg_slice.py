from wordle import create_wordle_response, SmartWordFilter, SimulatedGameState
from collections import defaultdict
from math import log
from joblib import Memory
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

memory = Memory(location='artifacts', verbose=0)

@memory.cache
def get_avg_slice(guess, solution_list):
    '''
    Returns the average "slice" of a word, which is how much, on average, 
    the word will cut the solution list.
    .e.g. if the word 'spams' has a slice of .8 for word list (....), 
    on average guessing the word 'spams' will cut the word list by 80%.
    '''
    assert len(solution_list) > 0

    response_frequency = defaultdict(lambda: 0)
    response_slice = defaultdict(lambda : {'avg_slice': 0, 'probability': None})

    count = float(len(solution_list))
    for hidden_word in solution_list:
        response = create_wordle_response(hidden_word=hidden_word, guess=guess)
        response_frequency[response] +=1

    for r,i in response_frequency.items():
        print(r.as_letters())  

    print(response_frequency)
   
    word_filter = SmartWordFilter(solution_list, legal_answers=solution_list)
    # hack, see below
    game = SimulatedGameState(tuple([guess]), solution_list, hidden_word='BLANK')
    for response, frequency in tqdm(response_frequency.items()):
        # hack
        game.guesses = list()
        game.add_guess(guess)
        for i, hidden_word in enumerate(solution_list):
            # this is a hack for performance
            game.responses = list()
            game.hidden_word = hidden_word
            #game = SimulatedGameState(tuple([guess]), solution_list, hidden_word=hidden_word)
            # normally we would create a new game for each, but thats a lot of games.
            game.add_response(response)
            possible_answers = word_filter.get_possible_words(game, sort=False, make_tuple=False)
            slice_perc = float(count - len(possible_answers))/count # The percentage of words this response would cut out.
            # sum up the slice percentage
            response_slice[response]['avg_slice'] += slice_perc
        # now calculate the average slice for this response
        response_slice[response]['avg_slice'] = response_slice[response]['avg_slice']/float(i)
        # and the probability
        response_slice[response]['probability'] = float(frequency)/count

    # Now return the probability weighted average of the average slice
    weighted_avg = 0
    n_responses = float(len(response_slice))
    for response, dic in response_slice.items():
        weighted_avg += dic['probability'] * dic['avg_slice']

    return guess, weighted_avg/ n_responses 



def get_all_slices(guess_list, solution_list, processes=1):
    results = [] # list of tuples of (word, slice)
    if processes == 1: # good for unit tests
        for guess in guess_list:
            results.append(get_entropy(guess, solution_list))
    else:
        with Pool(processes=processes) as pool:
            partial_avg_slice = partial(get_avg_slice, solution_list=solution_list)
            for result in pool.imap_unordered(partial_avg_slice, guess_list):
                results.append(result)
    return results
