from prompt_toolkit import prompt, PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
import click


def main():
    session = PromptSession(
            history=FileHistory('history.txt'),
                            auto_suggest=AutoSuggestFromHistory(),
            )
    welcome=u'Welcome to the Worlde Engine! Input your first guess to start: '
    user_input = session.prompt(welcome)

    while True:
        user_input = session.prompt(u'>')
        click.echo(user_input)

if __name__ == '__main__':
    main()
