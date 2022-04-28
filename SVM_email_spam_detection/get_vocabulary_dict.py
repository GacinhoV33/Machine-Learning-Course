#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv

from typing import Dict


def get_vocabulary_dict() -> Dict[int, str]:
    """Read the fixed vocabulary list from the datafile and return.

    :return: a dictionary of words mapped to their indexes
    """

    with open('data/vocab.txt', 'r') as data:
        vocab_dct = dict()
        for line in data.readlines():
            numb, word = line.split()
            vocab_dct[word] = int(numb)
    return vocab_dct
