## Overview
This repository contains a Python-based data analysis project. The project focuses on preprocessing, analyzing, and modeling text data collected from Reddit posts, particularly in German. It uses advanced natural language processing (NLP) techniques such as lemmatization, stopword removal, and topic modeling with BERTopic and TF-IDF.

---

## Features

- **Data Collection**:
  - Download of new data
  - Usage of already downloaded datasets (included: 20250316_151312_subreddit_posts_kassel.csv)
- **Data Preprocessing**:
  - Tokenization, lemmatization, and removal of stopwords.
  - Handling missing values and cleaning raw text data.
- **Topic Modeling**:
  - Using BERTopic with SentenceTransformer embeddings and TF-IDF vectorization.
  - Identifying key topics in German Reddit posts.
- **Visualization**:
  - Interactive topic visualizations using BERTopic.


---

## Installation

### Prerequisites
Ensure you have the following available:
- Python 3.8 or higher
- Redit API Key (register a useraccount on Reddit, then go to https://www.reddit.com/prefs/apps to obtain the API key)

### Clone the Repository
