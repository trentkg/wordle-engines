from wordle import GameState, WordleColor, WordleResponse, SmartWordFilter, WordCollection, SimulatedGameState

def test_filter():
    legal_words = ['women','nikau','swack','feens','fyles','poled','clags','starn','sharn','woops']
    hidden_word = 'sharn'
    game = GameState()
    word_filter = SmartWordFilter(legal_guesses = legal_words, legal_answers = legal_words)
    initial_set = set(word_filter.get_possible_words(game))
    first_expected_set = set(legal_words)
    assert initial_set == first_expected_set 
    
    # totally wrong
    game.add_guess('poled')
    response = WordleResponse(tuple(WordleColor.BLACK for x in range(5)))
    game.add_response(response)

    second_set = set(word_filter.get_possible_words(game))
    second_expected_set = first_expected_set - set(['poled', 'women', 'fyles', 'woops'])

    assert second_set == second_expected_set 

    # some yellows, no greens
    game.add_guess('nikau')
    response = WordleResponse([
        WordleColor.YELLOW,
        WordleColor.BLACK, 
        WordleColor.BLACK, 
        WordleColor.YELLOW,
        WordleColor.BLACK
        ])
    game.add_response(response)

    third_set = set(word_filter.get_possible_words(game))
    third_expected_set = set(filter(lambda x: 'n' in x and 'a' in x, second_expected_set - set(['nikau'])))
    
    assert third_set == third_expected_set 

    # some greens ! 
    game.add_guess('starn')
    response = WordleResponse([
        WordleColor.GREEN,
        WordleColor.BLACK, 
        WordleColor.GREEN, 
        WordleColor.GREEN,
        WordleColor.GREEN,
        ])
    game.add_response(response)

    fourth_set = set(word_filter.get_possible_words(game))
    fourth_expected_set = set(['sharn']) # the only answer

    assert fourth_set == fourth_expected_set


def test_filter_always_returns_set_doesnt_break_with_real_wordle_example():
    '''This real world example broke our filter today.'''
    game = GameState()
    words = WordCollection()
    word_filter = SmartWordFilter(legal_guesses=words.guesses, legal_answers=words.answers)
    # hidden word is  NASTY
        
    game.add_guess('calid')
    response = WordleResponse([
        WordleColor.BLACK, 
        WordleColor.GREEN,
        WordleColor.BLACK, 
        WordleColor.BLACK, 
        WordleColor.BLACK
        ])
    game.add_response(response)

    second_set = word_filter.get_possible_words(game)
    assert len(second_set) > 0

    game.add_guess('haven')
    response = WordleResponse([
        WordleColor.BLACK, 
        WordleColor.GREEN,
        WordleColor.BLACK, 
        WordleColor.BLACK,
        WordleColor.YELLOW
        ])
    game.add_response(response)

    third_set = word_filter.get_possible_words(game)
    assert len(third_set) > 0
    
    game.add_guess('fancy')
    response = WordleResponse([
        WordleColor.BLACK, 
        WordleColor.GREEN,
        WordleColor.YELLOW, 
        WordleColor.BLACK,
        WordleColor.GREEN,
        ])
    game.add_response(response)

    fourth_set = word_filter.get_possible_words(game)
    assert len(fourth_set) > 0

    game.add_guess('nappy')
    response = WordleResponse([
        WordleColor.GREEN, 
        WordleColor.GREEN,
        WordleColor.BLACK, 
        WordleColor.BLACK,
        WordleColor.GREEN,
        ])
    game.add_response(response)

    fifth_set = word_filter.get_possible_words(game)
    assert len(fifth_set) > 0

    game.add_guess('natty')
    response = WordleResponse([
        WordleColor.GREEN, 
        WordleColor.GREEN,
        WordleColor.BLACK, 
        WordleColor.GREEN,
        WordleColor.GREEN,
        ])
    game.add_response(response)

    sixth_set = word_filter.get_possible_words(game)
    assert len(sixth_set) > 0

    final_set = word_filter.get_possible_words(game)
    
    assert len(final_set) > 0

def test_filter_always_returns_set_doesnt_break_with_real_wordle_example():
    '''This real world example broke our filter today.'''
    game = GameState()
    words = WordCollection()
    word_filter = SmartWordFilter(legal_guesses=words.guesses, legal_answers=words.answers)
    # hidden word is  NASTY
        
    game.add_guess('calid')
    response = WordleResponse([
        WordleColor.BLACK, 
        WordleColor.GREEN,
        WordleColor.BLACK, 
        WordleColor.BLACK, 
        WordleColor.BLACK
        ])
    game.add_response(response)

    second_set = word_filter.get_possible_words(game)
    assert len(second_set) > 0

    game.add_guess('haven')
    response = WordleResponse([
        WordleColor.BLACK, 
        WordleColor.GREEN,
        WordleColor.BLACK, 
        WordleColor.BLACK,
        WordleColor.YELLOW
        ])
    game.add_response(response)

    third_set = word_filter.get_possible_words(game)
    assert len(third_set) > 0
    
    game.add_guess('fancy')
    response = WordleResponse([
        WordleColor.BLACK, 
        WordleColor.GREEN,
        WordleColor.YELLOW, 
        WordleColor.BLACK,
        WordleColor.GREEN,
        ])
    game.add_response(response)

    fourth_set = word_filter.get_possible_words(game)
    assert len(fourth_set) > 0

    game.add_guess('nappy')
    response = WordleResponse([
        WordleColor.GREEN, 
        WordleColor.GREEN,
        WordleColor.BLACK, 
        WordleColor.BLACK,
        WordleColor.GREEN,
        ])
    game.add_response(response)

    fifth_set = word_filter.get_possible_words(game)
    assert len(fifth_set) > 0

    game.add_guess('natty')
    response = WordleResponse([
        WordleColor.GREEN, 
        WordleColor.GREEN,
        WordleColor.BLACK, 
        WordleColor.GREEN,
        WordleColor.GREEN,
        ])
    game.add_response(response)

    sixth_set = word_filter.get_possible_words(game)
    assert len(sixth_set) > 0

    final_set = word_filter.get_possible_words(game)
    
    assert len(final_set) > 0
