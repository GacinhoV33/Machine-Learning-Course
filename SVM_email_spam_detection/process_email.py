#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
from typing import List

from nltk import PorterStemmer

from get_vocabulary_dict import get_vocabulary_dict


def process_email(email_contents: str) -> List[int]:
    """Pre-process the body of an email and return a list of indices of the
    words contained in the email.

    :param email_contents: the body of an email
    :return: a list of indices of the words contained in the email
    """
    vocabulary_dict = get_vocabulary_dict()

    word_indices = list()

    # ========================== Preprocess Email ===========================

    email_contents = email_contents.lower()

    # Strip all HTML
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)

    # Convert all sequences of digits (0-9) to a 'number' token.
    email_contents = re.sub('[0-9]+', 'number', email_contents)

    # Convert all strings starting with http:// or https:// to a 'httpaddr' token.
    email_contents = re.sub('http[s]*://[\S]+', 'httpaddr', email_contents)

    # Convert all strings with @ in the middle to a 'emailaddr' token.
    email_contents = re.sub('[\S]+@[\S]+', 'emailaddr', email_contents)

    email_contents = re.sub('[$]+', 'dollar', email_contents)

    # ========================== Tokenize Email ===========================

    # Output the email to screen as well
    print('\n==== Processed Email ====\n\n')
    print(email_contents)
    # Process file
    col = 0

    # Tokenize and also get rid of any punctuation
    tokens = re.split('[ @$/#.-:&*\+=\[\]?!\(\)\{\},''">_<;#\n\r]', email_contents)

    for token in tokens:

        # Remove any non alphanumeric characters
        token = re.sub('[^a-zA-Z0-9]', '', token)

        # Stem the word 
        token = PorterStemmer().stem(token.strip())

        # Skip the word if it is too short
        if len(token) < 1:
            continue

        for key, value in vocabulary_dict.items():
            if key == token:
                word_indices.append(value)

        if (col + len(token) + 1) > 78:
            print('')
            col = 0
        print('{} '.format(token), end='', flush=True)
        col = col + len(tokens) + 1

    # Print footer
    print('\n\n=========================\n')
    return word_indices
