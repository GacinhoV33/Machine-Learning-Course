#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import List

import numpy as np


def email_features(word_indices: List[int]) -> np.ndarray:
    """Convert a list of word IDs into a feature vector.

    :param word_indices: a list of word IDs
    :return: a feature vector from the word indices (a row vector)
    """

    # Total number of words in the dictionary
    n_words = 1899

    x = np.zeros(n_words)
    for ind in word_indices:
        x[ind-1] = 1
    return x.reshape(1, -1)