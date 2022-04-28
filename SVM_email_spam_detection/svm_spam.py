#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn import svm

from process_email import process_email
from email_features import email_features
from get_vocabulary_dict import get_vocabulary_dict


def read_file(file_path: str) -> str:
    """Return the content of the text file under the given path.

    :param file_path: path to the file
    :return: file content
    """
    with open(file_path, 'r') as file:
        return file.read()

# %% ==================== Part 1: Email Preprocessing ====================

print('\nPreprocessing sample email (emailSample1.txt)\n')

file_contents = read_file('data/emailSample1.txt')
word_indices = process_email(file_contents)

# Print Stats
print('Word Indices: \n')
print(word_indices)
print('\n\n')

# %% ==================== Part 2: Feature Extraction =====================
print('\nExtracting features from sample email (emailSample1.txt)\n')

# Extract Features
file_contents = read_file('data/emailSample1.txt')
word_indices = process_email(file_contents)
features = email_features(word_indices)

# Print Stats
print('Length of feature vector: {}\n'.format(len(features[0])))
print('Number of non-zero entries: {}\n'.format(sum(sum(features > 0))))

# input('Program paused. Press enter to continue.\n')

# %% =========== Part 3: Train Linear SVM for Spam Classification ========
#  In this section, you will train a linear classifier to determine if an
#  email is Spam or Not-Spam.

# Load the Spam Email dataset
# You will have X, y in your environment

print('\nLoading the training dataset...')
X_train = np.genfromtxt('data/spamTrain_X.csv', delimiter=',')
y_train = np.genfromtxt('data/spamTrain_y.csv', delimiter=',')
print('The training dataset was loaded.')

print('\nTraining Linear SVM (Spam Classification)\n')
print('(this may take 1 to 2 minutes) ...\n')

clf = svm.LinearSVC(C=0.1)
# clf = svm.SVC(kernel='linear', C=0.1, probability=True, random_state=1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_train)

acc_train = np.sum(y_train == y_pred)/y_train.shape[0]
print('Training Accuracy: {:.2f}%\n'.format(acc_train * 100))

# %% =================== Part 4: Test Spam Classification ================

print('\nLoading the test dataset...')
X_test = np.genfromtxt('data/spamTest_X.csv', delimiter=',')
y_test = np.genfromtxt('data/spamTest_y.csv', delimiter=',')
print('The test dataset was loaded.')

print('\nEvaluating the trained Linear SVM on a test set ...\n')

y_pred = clf.predict(X_test)

acc_test = np.sum(y_test==y_pred)/y_pred.shape[0]
print('Test Accuracy: {:.2f}%\n'.format(acc_test * 100))

# input('Program paused. Press enter to continue.\n')


# %% ================= Part 5: Top Predictors of Spam ====================

weights = clf.coef_.reshape(-1)
idx = np.argsort(-weights)

vocabulary_dict = get_vocabulary_dict()

print('\nTop predictors of spam: \n')
for i in range(15):
    print(' {word:<20}: {weight:10.6f}'.format(
        word=str([key for key, value in vocabulary_dict.items() if value == idx[i]][0]), weight=weights[idx[i]]))

print('\n\n')
filename = 'data/emailSample1.txt'

# Read and predict
file_contents = read_file(filename)
word_indices = process_email(file_contents)
x = email_features(word_indices)

y_pred = clf.predict(x)

print('\nProcessed {}\n\nSpam Classification: {}\n'.format(filename, y_pred[0] > 0))
print('(1 indicates spam, 0 indicates not spam)\n\n')
