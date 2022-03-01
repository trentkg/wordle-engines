#!venv/bin/python
import sys
import cmd
import click
import logging
import random
from enum import Enum
from english_words import english_words_lower_alpha_set


FORMAT = '%(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


       
class WordCollection:
    words = None 
    def __init__(self, style = 'all'):
        if not self.words:
            #if style == 'all':
            self.words = [x for x in english_words_lower_alpha_set if len(x) == 5]
            #elif style == 'easy':
            # elif style == 'hard:
            # else raise AssertionError

    def get_words(self):
        return self.words

class WordleResponse:
    def __init__(self, colors):
        assert len(colors) == 5, 'Not enough color blocks in wordle response! There must be 5.'
        self.colors = colors

    def is_correct(self):
        return all(x == WordleColor.Green for x in self.colors)
   

class InvalidGameStateError(RuntimeError):
    def __init__(self, message):
        super().__init__(message)


class GameState:
    max_rounds = 5

    def __init__(self):
        self.responses = list()
        self.guesses = list()
        self.current_possible_responses = list()

    def add_response(self, response):
        if len(self.responses) >= self.max_rounds:
            raise InvalidGameStateError("There are more than 5 wordle responses already!")
        self.responses.append(response)

    def add_guess(self, string):
        if len(self.guesses) >= self.max_rounds:
            raise InvalidGameStateError("There are more than 5 wordle guesses already!")
        self.guesses.append(string)

    def game_is_won(self):
        last_response = self.responses[len(self.responses) - 1]
        return last_response.is_correct() 

    def is_first_move(self):
        return len(self.guesses) == 0 

    def any_guesses_remaining(self): 
        return len(self.guesses) >= self.max_rounds

    def is_game_over(self):
        if self.is_first_move():
            return False

        if self.game_is_won():
            return True

        return self.any_guesses_remaining()

    def guess_number(self):
        return len(self.guesses)

class WordleColor(Enum):
    BLACK = 1
    YELLOW = 2
    GREEN = 3

class WordleAlgorithm:
    '''An algorithm that chooses the next guess for wordle'''
    def __init__(self, legal_words = WordCollection().get_words()):
        self.legal_words = legal_words

    def get_next_answer(self, game_state):
        valid_answers = self.get_possible_answers(game_state)
        return self.guess(valid_answers)

    def get_possible_answers(self, game_state):
        raise NotImplementedError()

    def guess(self, valid_answers):
        raise NotImplementedError()

class SimpleRandomWordleAlgorithm(WordleAlgorithm):
    '''Chooses a random word everytime, regardless of wordle's responses.'''
    
    def get_possible_answers(self, game_state):
        return self.legal_words

    def guess(self, valid_answers):
        return random.choice(valid_answers)

class PandasRandomWordleAlgorithm(SimpleRandomWordleAlgorithm):
    '''Chooses a random word everytime, but uses wordles responses to narrow its decision'''
    pass

class WordleCommandLineEngine(cmd.Cmd):
    intro = "Welcome to WordleEngine! Type 'help' or '?' to see a list of commands. 'Ctrl+c' to quit\n"
    prompt = '(wordle engine)'
    file = None

    def do_random(self, arg):
        '''Input your wordle guesses and use a wordle engine. 'random' would play wordle using the simplist engine, 
        one that uses totally random guesses on each run.'''
        RandomWordleCmdLoop().cmdloop()
        return False 

class WordleCmdLoop(cmd.Cmd):
    prompt = '(PROMPT)'

    def __init__(self, algorithm, game_state, name):
        super(WordleCmdLoop,self).__init__()
        self.algorithm = algorithm
        self.game_state = game_state
        self.name = name
    # we might instead implemented a blank algorithm

    def onecmd(self, line):
        # was this an  or 
        if line is None:
            return self.emptyline()
        if not self.game_state.is_game_over():
            colors = [WordleColor.BLACK for x in range(5)] 
            self.game_state.add_response(WordleResponse(colors))
            answer =  self.algorithm.get_next_answer(self.game_state)
            self.stdout.write(answer + '\n')
            stop = False
        else:
            stop = True
        return stop 

    def precmd(self, line):
        self.stdout.write("{}'s guess is:\n".format(self.name))
        return line

    def postcmd(self, stop, line):
        if self.game_state.is_game_over():
            self.stdout.write("Game is finished.")
            stop = True
            if self.game_state.game_is_won():
                self.stdout.write("We won! Number of guesses to solution was: {}".format(self.game_state.guess_number()))
            else:
                self.stdout.write("We lost! Bummer.")
        else:
            self.stdout.write("What was wordle's response?\n")
        return stop

class RandomWordleCmdLoop(WordleCmdLoop):
    prompt = '(random)'

    def __init__(self):
        super().__init__(SimpleRandomWordleAlgorithm(), GameState(), name='SimpleRandomEngine')


def main():
    try:
        WordleCommandLineEngine().cmdloop()
    except KeyboardInterrupt:
        logger.info('')
        logger.info('Thanks for playing!')
        sys.exit(0)
 
if __name__ == '__main__':
    main()
