import pytest
from ml.entropy import get_entropy
from wordle import WordCollection

def test_get_entropy():
    guess1 = 'brave'
    guess2 = 'pzazz'
    solution_list = ('bulge', 'brand', 'butte', 'bicep','basil', 'bused')
    _, entropy_of_guess1 = get_entropy(guess=guess1, solution_list=solution_list )
    _, entropy_of_guess2 = get_entropy(guess=guess2, solution_list=solution_list )
    # This should make intuitive sense, pzazz has lots of z's and therefore should be a low
    # entropy word for this solution list
    assert entropy_of_guess1 > entropy_of_guess2
