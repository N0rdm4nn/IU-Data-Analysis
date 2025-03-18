import praw
import pandas as pd
import datetime
import configparser
import re
import spacy
import nltk
from nltk.corpus import stopwords
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import HDBSCAN
from collections import Counter

# download necessary NLTK resources; uncomment to download
#nltk.download("stopwords")
#nltk.download("punkt")
#nltk.download("punkt_tab")


# for displaying/debugging purposes
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


##################################################################################################################
# 1) Establish connection ########################################################################################
##################################################################################################################


# Load credentials from config file (to keep them secure)
config = configparser.ConfigParser(interpolation=None)
config.read('config.ini')

# initialize Reddit client
reddit = praw.Reddit(
    client_id=config['RedditAPI']['client_id'],
    client_secret=config['RedditAPI']['client_secret'],
    user_agent=config['RedditAPI']['user_agent']
)

# define scraping parameter
subreddit = reddit.subreddit('kassel')
no_subs = 300


##################################################################################################################
# 2) Data collection #############################################################################################
##################################################################################################################


# collect as much data as possible, which could be used for various purposes
def fetch_reddit_data(subreddit):
    posts_data = []
    print(f"fetching data from r/{subreddit}")

    for submission in subreddit.new(limit=no_subs): #(others methods: controversial, gilded, hot, new, rising, top)

        # for more data, I also collect the comments of the submissions
        comments = [{
            "comment_id": comment.id,
            "parent_id": comment.parent_id,
            "author": str(comment.author),
            "body": comment.body,
            "created_utc": datetime.datetime.fromtimestamp(comment.created_utc),
            "score": comment.score
        } for comment in submission.comments.list()]

        # extract submission data
        post_data = {
            "author": str(submission.author),  # convert author to string to handle deleted accounts
            "title": submission.title,
            "text": submission.selftext,
            "created_utc": datetime.datetime.fromtimestamp(submission.created_utc),
            "score": submission.score, # number of upvotes
            "upvote_ratio": submission.upvote_ratio, # percentage of upvotes from all votes on the submission
            "num_comments": submission.num_comments,
            "id": submission.id,
            "author_flair_text": submission.author_flair_text if hasattr(submission, 'author_flair_text') else None,
            "url": submission.url,
            "comments": comments,
        }
        posts_data.append(post_data)

    # convert to DataFrame and save to CSV for further processing
    df = pd.DataFrame(posts_data)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_subreddit_posts_{subreddit}.csv"
    df.to_csv(filename, index=False)

    print(df.head())
    print(f"Saved {len(posts_data)} posts to {filename}")

    return df


##################################################################################################################
# 3) Pre-processing ##############################################################################################
##################################################################################################################


# either execute data collection; uncomment to start
#fetch_reddit_data(subreddit)

# or import raw data
raw_data = pd.read_csv("20250316_151312_subreddit_posts_kassel.csv") # modify filename for raw data input
print(raw_data.head())

# merge title and text and remove NaN values which may lead to error
raw_data["title"] = raw_data["title"].fillna("")
raw_data["text"] = raw_data["text"].fillna("")

# create a corpus for the topic analysis
raw_df = pd.DataFrame()
raw_df["raw"] = raw_data["title"] + " " + raw_data["text"]# + " " + raw_data["comments"]

# get stopwords for both English and German as submissions may be in both languages
stop_words_de = list(stopwords.words("german"))
stop_words_en = list(stopwords.words("english"))
stop_words =  stop_words_de + stop_words_en + ["hi", "hallo"]#list(stop_words_en.union(stop_words_de))

# use spaCy for lemmatization/tokenization of german language
nlp = spacy.load('de_core_news_sm')

def lemmatize_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ''
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if token.lemma_ != '-PRON-'] # filter pronouns

    return " ".join(lemmas)


# remove stopwords
def remove_stopwords(text):

    if isinstance(text, str):
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]

        return " ".join(filtered_words)

    return text


# remove unwanted characters and URLs
def preprocess(text):

    # convert to lowercase, remove non-alphabetic characters and URLs
    # die entsprechenden buchstaben äöü und sonderzeichen erfordert regexpressions
    text = re.sub(r"http\S+|www\S+|[^a-zA-ZäöüÄÖÜß\s]", "", text.lower())
    text= re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text#" ".join(text)


# apply preprocessing to raw data
raw_df["lemmatized"] = raw_df["raw"].apply(lambda x: " ".join([token.lemma_ for token in nlp(x)]))
raw_df["no_stopwords"] = raw_df["lemmatized"].apply(remove_stopwords)
raw_df["processed"] = raw_df["no_stopwords"].apply(preprocess)

# filter documents which are less relevant for sematic analysis
raw_df = raw_df[raw_df['processed'].str.split().str.len() > 5]
#print(raw_df.head(25))


##################################################################################################################
# 4) topic-modeling ##############################################################################################
##################################################################################################################


# use the preprocessed data to create the corpus
corpus = raw_df['processed'].tolist()

# load embedding model suitable for multilingual texts, alternative "paraphrase-multilingual-MiniLM-L12-v2"
embedding_model = SentenceTransformer(
    "distiluse-base-multilingual-cased-v1"
    #alternative "paraphrase-multilingual-MiniLM-L12-v2"
)
vectorizer_model = TfidfVectorizer(
    #stop_words=stop_words,   # remove German stopwords
    #min_df=2,              # ignore terms appearing in fewer than 2 documents
    #max_df=0.8,            # ignore terms appearing in more than 80% of documents
    ngram_range=(1, 2)      # consider both unigrams and bigrams
)

# initialiaze BERTopic model
topic_model = BERTopic(
    embedding_model=embedding_model, # represent the semantic meaning of the texts and enable the model to group
                                     # similar texts together
    vectorizer_model=vectorizer_model, # allows to specify ngrams, optional TfidfVectorizer
    language="multilingual",
    hdbscan_model=HDBSCAN(min_cluster_size=6),  # allow smaller clusters, to create topics
    calculate_probabilities=True,
    verbose=True
)


##################################################################################################################
# 5) topic analysis ##############################################################################################
##################################################################################################################


# train the model and perform topic modeling
topics, probabilities = topic_model.fit_transform(corpus)
print(topic_model.get_topic_info().head(6))

# output the top n topics
print("\nTop 5 topics:\n")
for topic_id in topic_model.get_topic_info()["Topic"].head(6):
    if topic_id == -1: # ignore outliers
        continue
    print(f"\nTopic {topic_id}:")
    for word in topic_model.get_topic(topic_id):
        print(word)

# add topic assignment in df
#raw_df['topic'] = topics
#print(raw_df.head(25))


##################################################################################################################
# 6) Entity analysis #############################################################################################
##################################################################################################################


# find most popular submissions by sorting the dataframe using the 'score' column in descending order and print
# top 5 submissions
sorted_df = raw_data.sort_values(by="score", ascending=False)
print("\nTop 5 submissions according to score:\n")
print(sorted_df[["score", "id", "author", "url", "title", "text"]][:5])

# find 5 most active users in the dataset
# extract the "author" column from raw data while removing deleted users
authors = Counter(raw_data["author"].dropna()).most_common(5)

print("\nTop 5 authors in the collected data:\n")
for author, count in authors:
    print(f"Author: {author}, Posts: {count}")

