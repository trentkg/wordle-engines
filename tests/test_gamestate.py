import pytest
from wordle import GameState, WordleResponse, WordleColor, InvalidGameStateError


@pytest.fixture
def game():
    return GameState()

def test_when_all_wrong(game):
    '''Test that the game ends when there are 6 rounds of guessing and all are wrong'''
    wrong = WordleResponse(colors=[WordleColor.BLACK for x in range(5)])
    for i in range(5):
        game.add_guess('guess')
        game.add_response(wrong)

    assert not game.game_over(), 'Game is over after 4 wrong guesses!'
    assert not game.is_won(), 'Game is won when every wordle response has been wrong!'
    assert game.any_rounds_remaining()

    game.add_guess('guess')
    game.add_response(wrong)
    assert game.game_over(), 'Game is NOT over when there have been 6 guesses!'
    assert not game.is_won(), 'Game is won when every guess has been wrong!'
    assert not game.any_rounds_remaining(), 'There are guesses remaining when there shouldn\'t be!'
    with pytest.raises(InvalidGameStateError) as excinfo:
        game.add_guess('guess')
    with pytest.raises(InvalidGameStateError) as excinfo:
        game.add_response(wrong)

def test_when_right(game):
    wrong = WordleResponse(colors=[WordleColor.BLACK for x in range(5)])
    game.add_guess('guess')
    game.add_response(wrong)

    wrong = WordleResponse(colors=[WordleColor.YELLOW for x in range(5)])
    game.add_guess('ascot')
    game.add_response(wrong)

    assert not game.game_over(), 'Game is over after only 2 wrong guesses!'
    assert not game.is_won(), 'Game is won when every wordle response has been wrong!'
    assert game.any_rounds_remaining(), 'No gueses remaining after 2 wrong guesses!'

    right = WordleResponse(colors=[WordleColor.GREEN for x in range(5)])
    game.add_guess('coats')
    game.add_response(right)


    assert game.game_over(), 'Game is not over after a correct guess!'
    assert game.is_won(), 'Game is not won after a correct guess!'
    assert game.any_rounds_remaining(), 'No guesse remaining after only 3 guesses!'

    with pytest.raises(InvalidGameStateError) as excinfo:
        game.add_guess('guess')
    with pytest.raises(InvalidGameStateError) as excinfo:
        game.add_response(wrong)

