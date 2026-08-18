"""DATA110 Lab 05 - Basic NLP text preprocessing practice."""
import re
import string

text = "Natural Language Processing helps computers work with human language! 2026."

# Lowercase
clean = text.lower()
# Remove numbers
clean = re.sub(r"\d+", "", clean)
# Remove punctuation
clean = clean.translate(str.maketrans("", "", string.punctuation))
# Tokenize at a simple word level
words = clean.split()

stop_words = {"the", "is", "a", "an", "and", "of", "to", "in"}
filtered = [w for w in words if w not in stop_words]

print("Original:", text)
print("Tokens:", words)
print("After stop-word removal:", filtered)

# Course concepts to pair with this code:
# corpus = collection of documents
# vocabulary = unique words/tokens
# BoW = word occurrence/frequency representation
# TF-IDF = weights words by frequency while down-weighting common words
# unigram/bigram/trigram = 1/2/3-word sequences
# stemming/lemmatization = reducing words toward a base/root form
