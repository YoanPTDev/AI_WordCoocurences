# AI - Word coocurences (CVM 2023)

## Introduction
This project is a Python application designed to extract, analyze, and visualize word co-occurrence relationships from text files. By utilizing several Python modules and features such as numpy, sqlite3, and object-oriented programming, this application generates a co-occurrence matrix for all words in the text and provides functionalities to interact with the generated data. 

## Getting Started

### Requirements
This application requires the following:
- Python 3.8 or higher
- SQLite3
- NumPy library
- codecs library
- argparse library

## Files in the Project

The project consists of several Python files, each serving different purposes:

- `entrainement.py`: This file contains the main class for text parsing and word co-occurrence matrix creation.
- `entrainement_bd.py`: It extends the `Entrainement` class to allow saving and loading word co-occurrence data from a SQLite database.
- `dao.py`: This is the Data Access Object (DAO) class, responsible for managing the SQLite database.
- `recherche.py`: It is used for searching the co-occurrence matrix for synonyms of a given word.
- `cluster.py`: This file is responsible for performing the K-means clustering on the co-occurrence data.
- `options.py`: This file defines the `Options` class to handle command-line arguments using `argparse`.

## How to Use
To use this project, you need to first train the application with a text file by running the `entrainement.py` script and providing the necessary arguments. This will generate a word co-occurrence matrix and save it to a SQLite database.

You can find UTF-8 encoded training text in the "textes" folder.

Once the training is done, you can use the `recherche.py` script to search for synonyms of a specific word. Additionally, you can use the `cluster.py` script to perform clustering on the co-occurrence data.

## Conclusion
This project provides an interesting way to analyze and visualize word co-occurrence relationships in texts. With various functionalities such as searching for synonyms and clustering, it provides an easy and effective way to explore and understand the context and relationships between words in a text. 

## Contact

