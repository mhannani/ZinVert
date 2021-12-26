from datetime import timedelta
from termcolor import colored

print(timedelta(seconds=60*60*24+1))

print(colored('The training process of the model took: ', 'green'), colored(f'{timedelta(seconds=60*60*24+1)}', 'red'))
