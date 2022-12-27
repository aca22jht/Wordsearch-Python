"""JASMINE HUI PING TAY
COM2004 (DATA-DRIVEN COMPUTING ASSIGNMENT)
version: v1.0
"""

from typing import List

import numpy as np
from utils import utils
from utils.utils import Puzzle
from difflib import SequenceMatcher
import scipy.linalg

# The required maximum number of dimensions for the feature vectors.
N_DIMENSIONS = 20


def load_puzzle_feature_vectors(image_dir: str, puzzles: List[Puzzle]) -> np.ndarray:
    """Extract raw feature vectors for each puzzle from images in the image_dir.

    OPTIONAL: ONLY REWRITE THIS FUNCTION IF YOU WANT TO REPLACE THE DEFAULT IMPLEMENTATION

    The raw feature vectors are just the pixel values of the images stored
    as vectors row by row. The code does a little bit of work to center the
    image region on the character and crop it to remove some of the background.

    You are free to replace this function with your own implementation but
    the implementation being called from utils.py should work fine. Look at
    the code in utils.py if you are interested to see how it works. Note, this
    will return feature vectors with more than 20 dimensions so you will
    still need to implement a suitable feature reduction method.

    Args:
        image_dir (str): Name of the directory where the puzzle images are stored.
        puzzle (dict): Puzzle metadata providing name and size of each puzzle.

    Returns:
        np.ndarray: The raw data matrix, i.e. rows of feature vectors.
    """

    return utils.load_puzzle_feature_vectors(image_dir, puzzles)


def compute_pca(train_data):
    """Function which computes the principal components (eigenvectors) of the covariance matrix by using the training data
    """
    covx = np.cov(train_data, rowvar = 0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals = (N - 20, N - 1))
    v = np.fliplr(v)

    return v


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduces the test data dimensions
    """

    # get the training data from the dictionary (model)
    train = np.asarray(model["fvectors_train"])

    # get the eigenvectors data from the dictionary (model)
    # function
    eigen = model["eigen"]
    pcatest_data = np.dot((data - np.mean(train)), eigen)


    return pcatest_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    # principal component axes are the 'eigenvectors' of the data's covariance matrix, compute the pca
    # and store them into 'eigen'
    eigen = compute_pca(fvectors_train)

    # reduce the training data dimensions
    pcatrain_data = np.dot((fvectors_train - np.mean(fvectors_train)), eigen)

    model = {}
    model["labels_train"] = labels_train.tolist()
    model["fvectors_train"] = fvectors_train.tolist()

    # store the eigen vectors into the dictionary
    model["eigen"] = eigen.tolist()

    #store the training data into the dictionary
    model["train"] = pcatrain_data.tolist()

    return model



def classify (train, train_label, test, features=None):
    """Uses the reduced dimensions of test and train data and classify using the Nearest Neighbour Classification.
    """

    train = np.asarray(train)
    train_labels = np.asarray(train_label)
    print(type(train))
    if features is None:
        features = np.arange(0, train.shape[1])


    train = train [:, features]
    test = test [:, features]

    # nearest neighbour
    x = np.dot (test, train.transpose())
    modtest = np.sqrt(np.sum(test*test, axis=1))
    modtrain = np.sqrt(np.sum(train*train, axis=1))
    dist = x/np.outer(modtest, modtrain.transpose())

    # cosine distance 
    nearest = np.argmax(dist, axis=1)
    mdist = np.max(dist, axis=1)
    label = train_labels[nearest]

    # print(len(label))

    return label



def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Reduced dimension of test and training data (pca train and pca test)

    Args:
        fvectors_train (np.ndarray): feature vectors that are to be classified, stored as rows.
        model (dict): a dictionary storing all the model parameters needed by your classifier.

    Returns:
        List[str]: A list of classifier labels, i.e. one label per input feature vector.
    """

    return classify(model["train"],model["labels_train"],fvectors_test)



def horizontal(labels, words):
    """Find the word from the list of words in 'words' horizontally on the 2-D grid (wordsearch). Accounts for both words that are forwards horizontally & backwards horizontally.
    """
    rows = labels.shape[0]
    columns = labels.shape[1]


    # horizontal matches
    matches = []

    for word in words:
        word = word.upper()
        found = False
        row = 0

        while row < rows and not found:
            col = 0
            while col < columns and not found:
                if word[0] == labels[row][col]:
                    word_length = len(word)

                    if word == ''.join(labels[row][col : col + word_length]):
                        matches.append((row, col, row, col + word_length - 1))
                        found = True

                    # backwards
                    elif word == (''.join(labels[row][col - word_length + 1: col + 1])[::-1]):
                        matches.append((row, col, row, col - word_length + 1))
                        found = True
                col += 1
            row += 1

        if not found:
            matches.append((-1, -1, -1, -1)) # skip over the word (word list lines up with the positions list)

    return(matches)


def vertical(labels, words):
    """Find the word from the list of words in 'words' vertically on the 2-D grid (wordsearch). Accounts for both words that are forwards vertically & backwards vertically."""
    rows = labels.shape[0]
    columns = labels.shape[1]

    matches = []

    for word in words:
        word = word.upper()
        found = False
        col = 0

        word_length = len(word)
        while col < columns and not found:
            row = 0
            while row <= rows - word_length and not found:
                if word[0] == labels[row][col]:

                    if word == ''.join(labels[row : row + word_length, col]): # doesn't try to index the subset first
                        matches.append((row, col, row + word_length - 1, col))
                        found = True

                    # find it vertically backwards
                    elif word == (''.join(labels[row - word_length + 1 : row + 1, col])[::-1]): # reverse [::-1]
                        matches.append((row, col, row - word_length + 1, col))
                        found = True

                row += 1
            col += 1
        if not found:
            matches.append((-1, -1, -1, -1)) # skip over the word (word list lines up with the positions list)

    return(matches)


def neg_diagonal(labels, words):
    """Find the word from the list of words in 'words' diagonally (negative) on the 2-D grid (wordsearch). Accounts for both words that are forwards diagonally (negative) & backwards diagonally                  (negative).
    """
    rows = labels.shape[0]
    columns = labels.shape[1]


    # negative diagonal matches
    matches = []

    for word in words:
        word = word.upper()
        found = False
        col = 0

        word_length = len(word)
        while col <= columns - word_length and not found:
            row = 0
            while row <= rows - word_length and not found:
                if word[0] == labels[row][col]:
                    try_word = ""
                    for i in range(word_length):
                        try_word += labels[row + i, col + i]

                        # Quality Purposes:
                        # see how many letters in both the words taken from words and the word found from the grid match & if > half then accept & proceed
                        matching_letter_count = sum([1 for c, c1 in zip(word, try_word) if c == c1])
                        half_of_letters_count = len(word) / 2

                    if matching_letter_count > half_of_letters_count: # here
                        matches.append((row, col, row + word_length - 1, col + word_length - 1))
                        found = True
                if not found:
                    word = word[::-1]
                    if word[0] == labels[row][col]:
                        try_word = ""
                        for i in range(word_length):
                            try_word += labels[row + i, col + i]

                        if word == try_word:
                            matches.append((row + word_length - 1, col + word_length - 1, row, col))
                            found = True
                    word = word[::-1]
                row += 1
            col += 1
        if not found:
            matches.append((-1, -1, -1, -1)) # skip over the word (word list lines up with the positions list)

    return(matches)


def pos_diagonal(labels, words):
    """Find the word from the list of words in 'words' diagonally (positive) on the 2-D grid (wordsearch). Accounts for both words that are forwards diagonally (positive) 
       & backwards diagonally (positive).
    """
    rows = labels.shape[0]
    columns = labels.shape[1]


    matches = []

    for word in words:
        word = word.upper()
        found = False
        word_length = len(word)
        col = 0

        while col <= columns - word_length and not found:
            row = rows - word_length
            while row < rows and not found:
                if word[0] == labels[row][col]:
                    try_word = ""
                    for i in range(word_length):
                        try_word += labels[row - i, col + i]

                    # Quality Purposes:
                    # see how many letters in both the words taken from words and the word found from the grid match & if > half then accept & proceed
                    matching_letter_count = sum([1 for c, c1 in zip(word, try_word) if c == c1])
                    half_of_letters_count = len(word) / 2

                    if matching_letter_count > half_of_letters_count:
                        matches.append((row, col, row - word_length + 1, col + word_length - 1))
                        found = True
                if not found:
                    word = word[::-1]
                    if word[0] == labels[row][col]:
                        try_word = ""
                        for i in range(word_length):
                            try_word += labels[row - i, col + i]

                        if word == try_word:
                            matches.append((row - word_length + 1, col + word_length - 1, row, col))
                            found = True
                    word = word[::-1]
                row += 1
            col += 1
        if not found:
            matches.append((-1, -1, -1, -1))

    return(matches)



def find_words(labels: np.ndarray, words: List[str], model: dict) -> List[tuple]:
    """Calls all the function used to find the words in 4 different directions and assigns them to a variable like (e.g: h, v, nd, pd)
       and performs them on the data & appends the positions of the words found nto the list of positions & the word_pos.append((0, 0, 1, 1)
       is to make the wordsearch fit nicely into the grid.)

        Args:
        labels (np.ndarray): 2-D array storing the character in each
            square of the wordsearch puzzle.
        words (list[str]): A list of words to find in the wordsearch puzzle.
        model (dict): The model parameters learned during training.

        Returns:
            list[tuple]: A list of four-element tuples indicating the word positions.
    """

    word_pos = []

    h = horizontal(labels,words)
    v = vertical(labels,words)
    nd = neg_diagonal(labels,words)
    pd = pos_diagonal(labels,words)
    for i in range(len(words)):
        if h[i] != (-1, -1, -1, -1):
            word_pos.append(h[i])
        elif v[i] != (-1, -1, -1, -1):
            word_pos.append(v[i])
        elif nd[i] != (-1, -1, -1, -1):
            word_pos.append(nd[i])
        elif pd[i] != (-1, -1, -1, -1):
            word_pos.append(pd[i])
        else: word_pos.append((0, 0, 1, 1))

    return word_pos

