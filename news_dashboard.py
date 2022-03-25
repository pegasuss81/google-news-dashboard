from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config
import pandas as pd
import numpy as np
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import nltk
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import gensim

nlp = spacy.load("en_core_web_sm", disable=['parser','ner','tagger', 'textcat'])
nltk.download('punkt')

user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent

def create_df(START, END, keyword):
    gn = GoogleNews(start=START, end=END)
    gn.search(keyword)
    for i in range(20):
        gn.getpage(i)
        result = gn.result()
        df = pd.DataFrame(result)
        #df.to_csv("data/google_news_{keyword}_{START}_{END}.csv")

    li1 = []
    for ind in df.index:
        d1 = {}
        article = Article(df["link"][ind])
        try:
            article.download()
            article.parse()
            article.nlp()
            d1['Date'] = df['date'][ind]
            d1['Datetime'] = df['datetime'][ind]
            d1['Media'] = df['media'][ind]
            d1['Title'] = article.title
            d1['Article'] = article.text
            d1['Summary'] = article.summary
            d1['URL'] = df['link'][ind]
        except:
            pass
        li1.append(d1)
    df_article = pd.DataFrame(li1)
    ## add columns for sentiment analysis
    df_article['polarity'] = df_article['Summary']
        .dropna()
        .apply(lambda text: nlp(' '.join(re.findall(r'\b\w+\b', text)))._.blob.polarity)
    df_article['sentiment'] = df_article['polarity'].apply(lambda x: "Positive" if x > 0.05 else "Negative" if x < -0.05 else "Neutral")
    df_article.to_csv("data/news_{keyword}_{START}_{END}.csv")
    return df_article


def customize_lemmatizer(doc):
    doc_cleaned = ' '.join(re.findall(r'\b\w[\w\']+\b', doc))
    return [ w.lemma_.lower() for w in nlp(doc_cleaned)
                       if w.lemma_ not in ['_', '.', '-PRON-'] ]


def customize_stopwords():
    stopwords = spacy.lang.en.stop_words.STOP_WORDS
    stopwords = stopwords.union({'ll', 've', 'pron', 'a', 'ukraine', 'ukrainian'})
    stopwords = set(customize_lemmatizer(' '.join(list(stopwords))))
    return stopwords


#df_article = create_df('Ukraine', '01/01/2022', '03/25/2022')
def process_words(df_article, stop_words=customize_stopwords()):
    result = []
    texts = df_article['Summary'].values.tolist()
    for t in texts:
        # t = ' '.join(re.findall(r'\b\w[\w\']+\b', t))
        try:
            t = ' '.join(re.findall(r'\b\w+\b', t))
            doc = nlp(t)
        except TypeError:
            pass
        result.append([token.lemma_.lower() for token in doc if token.lemma_ not in stop_words])
    return result

def create_LDA_model(df_article):
    num_topics = 5
    processed_text = process_words(df_article, stop_words=customize_stopwords().union(['-PRON-']))
    dictionary = gensim.corpora.Dictionary(processed_text)
    corpus = [dictionary.doc2bow(t) for t in processed_text]
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=dictionary,
                                                num_topics=num_topics,
                                                random_state=117, update_every=1,
                                                chunksize=1500,
                                                passes=5, iterations=10,
                                                alpha='asymmetric', eta=1 / 100,
                                                per_word_topics=True)

    return lda_model, corpus, processed_text


def get_main_topic_df(model, bow, texts, link, datetime, media, title):
    topic_list = []
    percent_list = []
    keyword_list = []

    for wc in bow:
        topic, percent = sorted(model.get_document_topics(wc), key=lambda x: x[1], reverse=True)[0]
        topic_list.append(topic)
        percent_list.append(round(percent, 3))
        keyword_list.append(' '.join(sorted([x[0] for x in model.show_topic(topic)])))

    result_df = pd.concat([pd.Series(topic_list, name='Dominant_topic'),
                           pd.Series(percent_list, name='Percent'),
                           pd.Series(texts, name='Processed_text'),
                           pd.Series(keyword_list, name='Keywords'),
                           pd.Series(link, name='Link'),
                           pd.Series(datetime, name='DateTime'),
                           pd.Series(media, name='Media'),
                           pd.Series(title, name='Title')
                           ]
                          , axis=1)

    return result_df

def get_representative_df(df_article, num_articles_per_topic):
    lda_model, corpus, processed_text = create_LDA_model(df_article)
    link = df_article['URL'].values.tolist()
    datetime = df_article['Datetime'].values.tolist()
    media = df_article['Media'].values.tolist()
    title = df_article['Title'].values.tolist()

    main_topic_df = get_main_topic_df(lda_model, corpus, processed_text, link, datetime, media, title)
    grouped_topics = main_topic_df.groupby('Dominant_topic')
    representatives = pd.DataFrame()

    for k in grouped_topics.groups.keys():
        representatives = pd.concat([representatives,
                        grouped_topics.get_group(k).sort_values(['Percent'], ascending=False).head(num_articles_per_topic)], ignore_index=True)

    return representatives

