from flask import Flask, render_template, request
import os
import nltk
import pke
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
import random
import re
import requests
from pywsd.similarity import max_similarity
from pywsd.lesk import adapted_lesk
from nltk.corpus import wordnet as wn
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('popular')

app = Flask(__name__)

# Function to generate Word Cloud
def generateWordCloud(keywords):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(keywords))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('static/wordcloud.png')
    plt.close()

# Function to plot Sentence Length Distribution
def plotSentenceLengthDistribution(sentences):
    lengths = [len(sent.split()) for sent in sentences]
    plt.figure(figsize=(10, 6))
    sns.histplot(lengths, kde=True, bins=20, color='blue')
    plt.title('Sentence Length Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.savefig('static/sentence_length_distribution.png')
    plt.close()

# Function to extract important words
def getImportantWords(art): 
    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input=art, language='en')
    pos = {'PROPN'}
    stops = list(string.punctuation) 
    stops += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    stops += nltk.corpus.stopwords.words('english')
    extractor.candidate_selection(pos=pos)
    extractor.candidate_weighting()
    result = [] 
    ex = extractor.get_n_best(n=25)
    for each in ex:
        result.append(each[0]) 
    return result

# Function to split text into sentences
def splitTextToSents(art):
    s = [sent_tokenize(art)]
    s = [y for x in s for y in x]
    s = [sent.strip() for sent in s if len(sent) > 15]
    return s

# Function to map important words to sentences
def mapSents(impWords, sents):
    processor = KeywordProcessor()
    keySents = {}
    for word in impWords:
        keySents[word] = []
        processor.add_keyword(word)
    for sent in sents:
        found = processor.extract_keywords(sent)
        for each in found:
            keySents[each].append(sent)
    for key in keySents.keys():
        temp = keySents[key]
        temp = sorted(temp, key=len, reverse=True)
        keySents[key] = temp
    return keySents

# Function to get word sense using WSD
def getWordSense(sent, word):
    word = word.lower()
    if len(word.split()) > 0:
        word = word.replace(" ", "_")
    synsets = wn.synsets(word, 'n')
    if synsets:
        wup = max_similarity(sent, word, 'wup', pos='n')
        adapted_lesk_output = adapted_lesk(sent, word, pos='n')
        lowest_index = min(synsets.index(wup), synsets.index(adapted_lesk_output))
        return synsets[lowest_index]
    else:
        return None

# Function to get distractors from WordNet
def getDistractors(syn, word):
    dists = []
    word = word.lower()
    actword = word
    if len(word.split()) > 0:
        word.replace(" ", "_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0:
        return dists
    for each in hypernym[0].hyponyms():
        name = each.lemmas()[0].name()
        if name == actword:
            continue
        name = name.replace("_", " ")
        name = " ".join(w.capitalize() for w in name.split())
        if name and name not in dists:
            dists.append(name)
    return dists

# Function to get distractors using ConceptNet API
def getDistractors2(word):
    word = word.lower()
    actword = word
    if len(word.split()) > 0:
        word = word.replace(" ", "_")
    dists = []
    url = f"http://api.conceptnet.io/query?node=/c/en/{word}/n&rel=/r/PartOf&start=/c/en/{word}&limit=5"
    obj = requests.get(url).json()
    for edge in obj['edges']:
        link = edge['end']['term']
        url2 = f"http://api.conceptnet.io/query?node={link}&rel=/r/PartOf&end={link}&limit=10"
        obj2 = requests.get(url2).json()
        for edge in obj2['edges']:
            word2 = edge['start']['label']
            if word2 not in dists and actword.lower() not in word2.lower():
                dists.append(word2)
    return dists

# Function to generate MCQs from the input text
def generateMCQs(text):
    impWords = getImportantWords(text)
    generateWordCloud(impWords)  # Generate Word Cloud
    sents = splitTextToSents(text)
    plotSentenceLengthDistribution(sents)  # Plot Sentence Length Distribution
    mappedSents = mapSents(impWords, sents)
    mappedDists = {}

    for each in mappedSents:
        wordsense = getWordSense(mappedSents[each][0], each)
        if wordsense:
            dists = getDistractors(wordsense, each)
            if len(dists) == 0:
                dists = getDistractors2(each)
            if len(dists) != 0:
                mappedDists[each] = dists
        else:
            dists = getDistractors2(each)
            if len(dists) > 0:
                mappedDists[each] = dists

    mcq_result = []
    for each in mappedDists:
        sent = mappedSents[each][0]
        p = re.compile(each, re.IGNORECASE)
        op = p.sub("________", sent)
        options = [each.capitalize()] + mappedDists[each]
        options = options[:4]
        opts = ['a', 'b', 'c', 'd']
        random.shuffle(options)
        mcq = {
            "question": op,
            "options": {opts[i]: options[i] for i in range(len(options))},
        }
        mcq_result.append(mcq)

    return mcq_result

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["file"]
        text = file.read().decode("utf-8")
        mcqs = generateMCQs(text)
        return render_template("index.html", mcqs=mcqs)
    return render_template("index.html", mcqs=[])

if __name__ == "__main__":
    app.run(debug=True)
