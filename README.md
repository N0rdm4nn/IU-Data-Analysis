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
- **Entity Analysis**:
  - Representation of top submissions according to score
  - Representation of most active authors



---

## Installation

### Prerequisites
Ensure you have the following available:
- Python 3.8 or higher
- Redit API Key (register a useraccount on Reddit, then go to https://www.reddit.com/prefs/apps to obtain the API key)
- Install the German language model (de_core_news_md) for the spaCy library 
```python -m spacy download de_core_news_sm```

### Clone the Repository
- install requirements: pip install- -r requirements.txt
  - sometimes there occurs an error when installing HDBSCAN. If this happens, additionally install following packages
    ```apt-get install python3.12-dev build essential```

### Run the code
- either use the provided sample data and run RedditBERTopicTfidf.py
- OR if a Reddit API Key is available:
  - insert credentials into the config.ini
  - choose a subreddit and number of submissions that should be downloaded (lines 44 & 45)
  - uncomment line 104 and comment line 107

