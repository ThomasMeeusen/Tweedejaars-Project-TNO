"""
Labeling

@author: Erik Meijer

Dit script wordt gebruikt om de dataset met nieuwsartikelen van Huggingface
te labelen. Het script leest een CSV bestand gemaakt met `export.ipnb` in.
Hetzelfde bestand wordt overscheven.

Run het script in de terminal om te labelen.

"""

# vul hier de locatie van het CSV bestand in
filename = "divided2/0-503.csv"

import pandas as pd
import numpy as np
import os
import nltk
import itertools

class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

def repr_article(article, highlights, encoding, id, label):
    wipe_screen()
    print(bcolors.HEADER + highlights + bcolors.ENDC)
    print()
    words = article.split()


    max_lines = 20
    lines = 0
    max_line_length = 80
    line_length = 0

    for word in words:
        print(word, end=' ')
        line_length += len(word) + 1
        if line_length > max_line_length:
            lines += 1
            line_length = 0
            print()
        if lines > max_lines:
            print('\n...')
            break

    print()

def wipe_screen():
    os.system("cls" if os.name == "nt" else "clear")

def main():
    df = pd.read_csv(filename, index_col=0)

    unlabeled_rows = df.index[df['label'].isnull()].tolist()
    changes_made = False

    while len(unlabeled_rows) >= 1:
        index = unlabeled_rows[0]

        # print the article in the terminal window
        repr_article(*df.loc[index].values.flatten().tolist())

        # ask user for input
        answer = input('enter 0 to label as NOT WORKPLACE, 1 to label as WORKPLACE, enter anything else to quit >')
        if answer == '0' or answer == '1':
            # add label to dataframe
            df['label'][index] = int(answer)
            changes_made = True
            unlabeled_rows.pop(0)
        else:
            break

    wipe_screen()
    if changes_made:
        df.to_csv(filename, index=True)
        print('Written to:', filename)

    print(np.sum(df['label'] == 0), 'of', len(df), 'rows labeled 0')
    print(np.sum(df['label'] == 1), 'of', len(df), 'rows labeled 1')
    print(np.sum(df['label'].isnull()), 'of', len(df), 'rows unlabeled')

if __name__ == "__main__":
    main()
