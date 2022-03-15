#!venv/bin/python
import sys
import cmd
import click
import logging
import random
import colorama
import ml
import gym
import gym_wordle
from pprint import pformat
from enum import Enum
from english_words import english_words_lower_alpha_set
from time import sleep

FORMAT = '%(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

colorama.init(autoreset=True)

### Make this a singleton       
class WordCollection:
    '''Responsible for building the collection legal words and legal answers.'''
    guesses = None
    answers = None

    def __init__(self, _type = 'wordle-words'):
        if  _type == 'all-english-words':
            self.guesses = [x for x in english_words_lower_alpha_set if len(x) == 5]
            self.answers = self.guesses
        elif _type == 'wordle-words':
            self.guesses = list()
            with open('artifacts/possible-wordle-answers.txt') as f1:
                for line in f1:
                    word = line.strip().lower()
                    if len(word) == 5:
                        self.guesses.append(word)
            self.answers = list()
            with open('artifacts/all-possible-wordle-answers-easy.txt') as f2:
                for line in f2:
                    word = line.strip().lower()
                    if len(word) == 5:
                        self.answers.append(word)
        else: 
            raise ValueError("Unknown type of word collection: {}".format(_type)) 

class WordleResponse:
    def __init__(self, colors):
        assert len(colors) == 5, 'Not enough color blocks in wordle response! There must be 5.'
        self.colors = colors

    def is_correct(self):
        return all(x == WordleColor.GREEN for x in self.colors)

    def __repr__(self):
        return pformat(self.colors) 
   

class InvalidGameStateError(RuntimeError):
    def __init__(self, message):
        super().__init__(message)


class GameState:
    max_rounds = 6

    def __init__(self):
        self.responses = list()
        self.guesses = list()

    def reset(self):
        self.responses = list()
        self.guesses = list()

    def __repr__(self):
        lines = list()
        space = '-----------'
        lines.append(space)
        colorama.reinit()
        for word, response in zip(self.guesses, self.responses):
            line = ''
            for letter, color in zip(word, response.colors):
                background = None
                if color == WordleColor.GREEN:
                    background = colorama.Back.GREEN
                elif color == WordleColor.YELLOW:
                    background = colorama.Back.YELLOW 
                elif color == WordleColor.BLACK:
                    background = colorama.Back.BLACK 
                else:
                    raise RuntimeError("Unknown wordle color: " + str(color))
                line += colorama.Style.BRIGHT + background + colorama.Fore.WHITE + letter 


            lines.append('|  '+line+colorama.Style.RESET_ALL+'  |' )
        lines.append(space)
        return '\n'.join(lines)

    def add_response(self, response):
        if len(self.responses) >= self.max_rounds:
            raise InvalidGameStateError("There are more than 6 wordle responses already!")
        if self.game_over():
            raise InvalidGameStateError("Cannot add a wordle response - game is over!")

        self.responses.append(response)

    def add_guess(self, string):
        if len(self.guesses) >= self.max_rounds:
            raise InvalidGameStateError("There are more than 6 wordle guesses already!")
        if self.game_over():
            raise InvalidGameStateError("Cannot add another guess - game is over!")

        self.guesses.append(string)

    def is_won(self):
        if len(self.responses) == 0:
            return False
        last_response = self.responses[len(self.responses) - 1]
        return last_response.is_correct() 

    def is_first_move(self):
        return len(self.guesses) == 0 

    def any_rounds_remaining(self): 
        return self.max_rounds > len(self.guesses) or \
                self.max_rounds > len(self.responses)

    def guesses_remaining(self):
        return self.max_rounds - len(self.guesses)

    def game_over(self):
        if self.is_first_move():
            return False

        if self.is_won():
            return True

        return not self.any_rounds_remaining()

    def guess_number(self):
        return len(self.guesses)

class SimulatedGameState(GameState):
    def __init__(self, legal_guesses, legal_answers):
        super().__init__()
        self.legal_guesses = legal_guesses
        self.legal_answers = legal_answers
        self.hidden_word = random.choice(self.legal_answers)

    def simulate_response(self, guess):
        colors = list()
        for index, letter in enumerate(guess):
            if letter == self.hidden_word[index]:
                colors.append(WordleColor.GREEN)
            elif letter in self.hidden_word:
                colors.append(WordleColor.YELLOW)
            else:
                colors.append(WordleColor.BLACK)
        return WordleResponse(colors)

class BadFormatError(Exception):
    '''Raised when an answer to a prompt is poorly formatted.'''
    pass

class WordleColor(Enum):
    BLACK = 1
    YELLOW = 2
    GREEN = 3

    @classmethod
    def from_letter_inputs(cls):
        return ('b', 'y', 'g')

    @classmethod
    def from_letter(cls, string):
        if string.lower() == 'b':
            return WordleColor.BLACK
        elif string.lower() == 'y':
            return WordleColor.YELLOW
        elif string.lower() == 'g':
            return WordleColor.GREEN
        else:
            raise BadFormatError('Letter is not b, y, or g!')

class WordFilter:
    def __init__(self, legal_guesses, legal_answers):
        self.legal_guesses = legal_guesses
        self.legal_answers = legal_answers

    def get_possible_words(self, game_state):
        raise NotImplementedError()

class BasicWordFilter(WordFilter):
    '''Returns the entire set of legal words every time.'''
    def get_possible_words(self, game_state):
        return self.legal_guesses

class SmartWordFilter(WordFilter):
    '''Filters possible answers to wordle using previous wordle responses'''

    def get_possible_words(self, game_state):
        '''
        Get the set of all valid answers to wordle given the state of the game.
        Written to be easy to understand and accurate, not by any means the most efficient!
        '''
        assert len(game_state.responses) == len(game_state.guesses), "Games responses and guesses are not equal! GameState is: {}".format(str(game_state))

        if game_state.is_first_move():
            return self.legal_guesses
       
        # words that pass all of the conditions in wordles responses
        valid_words = list()

        for word in self.legal_answers:
            if self._is_valid_word(game_state.responses, game_state.guesses, word):
                valid_words.append(word)
        
        assert len(valid_words) > 0, 'No valid words found!'
        return valid_words 

    def _is_valid_word(self, responses, guesses, word):
        for wordle_response, guess in zip(responses, guesses):
            if not self._passes_wordle_response(wordle_response, guess, word):
                return False
        return True

    def _passes_wordle_response(self, wordle_response, guess, word):
        '''Checks if a word passes the conditions in WordleResponse'''

        for index, color in enumerate(wordle_response.colors):
            guess_letter = guess[index]
            word_letter = word[index]
            if color == WordleColor.BLACK and guess_letter == word_letter:
                return False
            elif color == WordleColor.YELLOW:
                if guess_letter == word_letter:
                    return False
                elif guess_letter not in word:
                    return False
            elif color == WordleColor.GREEN and guess_letter != word_letter:
                return False
        return True 


class WordleAlgorithm:
    '''An algorithm that chooses the next guess for wordle'''
    word_filter_class = WordFilter

    def __init__(self, legal_guesses = None, legal_answers = None ):
        words = None
        if legal_guesses is None or legal_answers is None: 
            words = WordCollection()
        if legal_guesses is None:
            legal_guesses = words.guesses
        if legal_answers is None:
            legal_answers = words.answers

        self.legal_answers = legal_answers
        self.legal_guesses = legal_guesses
        self.word_filter = self.word_filter_class(legal_guesses, legal_answers)

    def get_next_answer(self, game_state):
        valid_answers = self.get_possible_answers(game_state)
        return self.guess(valid_answers, self.legal_guesses, game_state)

    def get_possible_answers(self, game_state):
        return self.word_filter.get_possible_words(game_state)

    def guess(self, valid_answers, valid_guesses, game_state):
        raise NotImplementedError()

class SimpleRandomWordleAlgorithm(WordleAlgorithm):
    '''Chooses a random valid answer everytime, regardless of wordle's responses.'''
    word_filter_class = BasicWordFilter

    def guess(self, valid_answers, valid_guesses, game_state):
        return random.choice(valid_answers)


class QLearningWordleAlgorithm(WordleAlgorithm):
    '''Chooses an answer based on its qLearning model'''
    word_filter_class = BasicWordFilter

    def __init__(self, state_file='artifacts/qlearning.pkl'):
        super().__init__()
        self.state_file = state_file
        self.qlearn = ml.core.QLearningAlgorithm(state_file=state_file)

    def guess(self, valid_answers, valid_guesses, game_state):
        env = self._map_game_state(game_state)
        return self.qlearn.predict(env)

    def _map_game_state(self, game_state):
        '''Transforms the game state np array that mirrors the 
        board in gym.make("Wordle-v0")'''
        for guess, response in zip(game_state.guesses, game_state.responses):
            raise RuntimeError("you haven't finished this yet")

        return None

class RandomWordleAlgorithm(SimpleRandomWordleAlgorithm):
    '''Chooses a random valid answer everytime, but uses wordles responses to narrow its decision'''
    word_filter_class = SmartWordFilter

class SaletsWordleAlgorithm(WordleAlgorithm):
    '''
    Uses a hardcoded first guess.
    '''
    word_filter_class = SmartWordFilter

    def guess(self, valid_answers, valid_guesses, game_state):
        remaining_guesses = game_state.guesses_remaining()
        nrounds = game_state.max_rounds
        this_guess = None

        if remaining_guesses == 6:
            # Some mathemtician online found this to be the best first guess
            this_guess = 'salet'
        else: 
            this_guess = random.choice(valid_answers)

        return this_guess

class ReinforcementLearningWordleAlgorithm(WordleAlgorithm):
    '''Uses reinforcement learning to find the best way to choose a word.'''
    pass


class WordleMenu(cmd.Cmd):
    intro = "Welcome to WordleEngine! Type 'help' or '?' to see a list of worlde engines. Type the engine name to use an engine to play. Type the engine name plus " + \
    "'simulate <Num Games>' to watch the engine play <Num Games>. Ctrl+c' to quit.\n"
    prompt = '(menu)'
    file = None

    def do_qlearning(self, arg):
        '''Use a simple qlearning algorithm'''
        args = arg.split()
        kwargs = { 
                'name': 'QLearningAlgo', 
                'prompt': '(qlearn-engine)'
                }
        if len(args):
            cmd = args[0] 
            if  cmd == 'simulate':
                num_games = int(args[1])
                kwargs['num_games'] = num_games
                kwargs['algorithm'] : QLearningWordleAlgorithm()
                SimulatedCmdLoop(**kwargs).cmdloop()
            elif cmd == 'train':
                num_games = int(args[1])
                kwargs['num_games'] = num_games
                kwargs['algorithm'] = ml.core.QLearningAlgorithm
                TrainingCmdLoop(**kwargs).cmdloop()

            else:
                kwargs['game_state'] = GameState()
                kwargs['algorithm'] : QLearningWordleAlgorithm()

                ManualCmdLoop(**kwargs).cmdloop()

        return False 



    def do_simplerandom(self, arg):
        '''Use the simplist engine, 
        one that uses totally random guesses on each run, regardless of the previous response.'''
        args = arg.split()
        kwargs = {'algorithm': SimpleRandomWordleAlgorithm(), 
                'name': 'SimpleRandomAlgo', 
                'prompt': '(rand-simple-engine)'
                }
        if len(args):
            if args[0] == 'simulate':
                num_games = int(args[1])
                kwargs['_type'] = 'simulate'
                kwargs['num_games'] = num_games
                kwargs['num_games'] = num_games
                SimulatedCmdLoop(**kwargs).cmdloop()
        else:
            kwargs['game_state'] = GameState()
            ManualCmdLoop(**kwargs).cmdloop()

        return False 

    def do_random(self, arg):
        '''Use a random engine that filters the possible wordset using wordle\'s previous responses.
        '''
        args = arg.split()
        kwargs = {'algorithm': RandomWordleAlgorithm(), 
                'name': 'RandomAlgo', 
                'prompt': '(rand-engine)'
                }
        if len(args):
            if args[0] == 'simulate':
                num_games = int(args[1])
                kwargs['num_games'] = num_games
                SimulatedCmdLoop(**kwargs).cmdloop()
        else:
            kwargs['game_state'] = GameState()
            ManualCmdLoop(**kwargs).cmdloop()

        return False 

    def do_salet(self, arg):
        '''Use an algorithm with a hardcoded first guess ('salet') 
        '''
        args = arg.split()
        kwargs = {'algorithm': SaletsWordleAlgorithm(), 
                'name': 'SaletAlgo', 
                'prompt': '(salet-engine)'
                }
        if len(args):
            if args[0] == 'simulate':
                num_games = int(args[1])
                kwargs['num_games'] = num_games
                SimulatedCmdLoop(**kwargs).cmdloop()
        else:
            kwargs['game_state'] = GameState()
            ManualCmdLoop(**kwargs).cmdloop()

        return False 

class SimulatedCmdLoop(cmd.Cmd):
    def __init__(self, algorithm, name, prompt, num_games):
        super().__init__()
        self.algorithm = algorithm
        self.name = name
        self.prompt = prompt
        self.num_games = num_games

    def preloop(self):
        self.stdout.write('Lets watch the simulation! {} games will play.\n'.format(self.num_games))
        self.stdout.write('Hit enter to start simulating. \n')

    def onecmd(self, line):
        stop = False
        games_left = self.num_games

        words = WordCollection()

        game = SimulatedGameState(legal_guesses = words.guesses, legal_answers = words.answers)

        while games_left >= 1:
            while not game.game_over():
                guess =  self.algorithm.get_next_answer(game)
                game.add_guess(guess)
                response = game.simulate_response(guess)
                game.add_response(response)
                self.stdout.write(str(game) + '\n\n')
                sleep(2)
                
            games_left -= 1
            
            self.stdout.write("Game is finished! {} game(s) left to simulate.\n".format(games_left))
            if game.is_won():
                self.stdout.write("We won! Number of guesses to solution was: {}\n".format(game.guess_number()))
            else:
                self.stdout.write("We lost! Bummer. \n")
            game.reset()
            sleep(5)

        stop = True
        return stop

class TrainingCmdLoop(cmd.Cmd):
    def __init__(self, algorithm, name, prompt, num_games):
        super().__init__()
        self.algorithm = algorithm
        self.name = name
        self.prompt = prompt
        self.num_games = num_games

    def preloop(self):
        self.stdout.write('Training started! {} games will play.\n'.format(self.num_games))

    def onecmd(self, line):
        self.algorithm.train(num_episodes=self.num_games)
        stop = True
        return stop


class ManualCmdLoop(cmd.Cmd):
    prompt = None 

    def __init__(self, algorithm, game_state, name, prompt):
        super().__init__()
        self.algorithm = algorithm
        self.game = game_state
        self.name = name
        self.prompt = prompt

    def preloop(self):
        if self.game.is_first_move(): # this should always be true
            first_guess = self.algorithm.get_next_answer(self.game)
            self.stdout.write('Lets play! {}\'s first guess is: '.format(self.name) + '\n')
            self.stdout.write(first_guess + '\n')
            self.game.add_guess(first_guess)
            self.stdout.write("What is wordle's response?\n")

    def get_answer(self, line):
        if len(line) != 5:
            raise BadFormatError("Line must be 5 characters long, one for each wordle square.")
        for letter in line:
            if letter.lower() not in WordleColor.from_letter_inputs():
                raise BadFormatError('''Each character in the line must be either "b" for 
                        "black", "y" for "yellow", or "g" for "green"''')
        return line
                
    def onecmd(self, line):
        stop = False
        if line is None:
            return self.emptyline()
        try:
            answer = self.get_answer(line)
        except BadFormatError as e:
            self.stdout.write(e.args[0])
            self.answered = False
            return stop

        # grab the response
        colors = [WordleColor.from_letter(x) for x in answer] 
        self.game.add_response(WordleResponse(colors))
        self.stdout.write("Got it!\n")
        self.stdout.write("The board should look like:\n")
        self.stdout.write(str(self.game)+'\n')
            # Then give the next guess
        if not self.game.game_over():
            next_guess =  self.algorithm.get_next_answer(self.game)
            self.stdout.write("{}'s next guess is:\n".format(self.name))
            self.stdout.write(next_guess + '\n')
            self.game.add_guess(next_guess)
        else:
            stop = True

        return stop 

    
    def postcmd(self, stop, line):
        if self.game.game_over():
            stop = True
        else:
            self.stdout.write("What is wordle's response?\n")
        return stop

    def postloop(self):
        if self.game.game_over():
            self.stdout.write("Game is finished.")
            stop = True
            if self.game.is_won():
                self.stdout.write("We won! Number of guesses to solution was: {}\n".format(self.game.guess_number()))
            else:
                self.stdout.write("We lost! Bummer. Back to main menu!\n")
        else:
            self.stdout.write("What was wordle's response?\n")
        return stop

def main():
    try:
        WordleMenu().cmdloop()
    except KeyboardInterrupt:
        logger.info('')
        logger.info('Thanks for playing!')
        sys.exit(0)
 
if __name__ == '__main__':
    main()
