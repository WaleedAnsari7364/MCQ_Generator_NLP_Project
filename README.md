MCQ Generator from Text
This project is a Flask-based web application that processes a given text to generate multiple-choice questions (MCQs). The system performs several text processing tasks such as keyword extraction, word sense disambiguation (WSD), and generating distractor words. Additionally, it visualizes data like word clouds and sentence length distributions to provide insights into the input text.

Features
MCQ Generation:

Extracts important keywords from the provided text.

Generates multiple-choice questions by creating a blank space (_______) for the extracted keywords.

Provides distractor words using WordNet and ConceptNet APIs.

Word Cloud Generation:

Visualizes the important keywords as a word cloud image, which is saved and displayed on the page.

Sentence Length Distribution:

Visualizes sentence lengths using a histogram to show the distribution of sentence word counts.

Text Processing:

The text is split into sentences and important words are extracted using the MultipartiteRank algorithm.

Uses the Lesk algorithm and WordNet to disambiguate word senses and find related words (distractors).

Interactive Web Interface:

A simple Flask web application where users can upload a text file and get MCQs along with visualizations.

Technologies Used
Flask: Web framework to create the web application.

NLTK: Natural language processing library for tasks like tokenization, stopword removal, and word sense disambiguation.

PKE: Python package for keyword extraction using unsupervised learning.

FlashText: Used for fast keyword extraction and mapping sentences with important words.

WordCloud: Library to generate word clouds from keywords.

Matplotlib & Seaborn: For plotting the sentence length distribution and other visualizations.

PyWSDS: Used for word sense disambiguation and similarity calculations.
