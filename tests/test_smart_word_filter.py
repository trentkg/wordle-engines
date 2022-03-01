from wordle import GameState, WordleColor, WordleResponse, SmartWordFilter

def test_filter():
    legal_words = ['women','nikau','swack','feens','fyles','poled','clags','starn','sharn','woops']
    hidden_word = 'sharn'
    game = GameState()
    word_filter = SmartWordFilter(legal_words = legal_words)
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








