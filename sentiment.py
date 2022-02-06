import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def calc_scores(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    return sentiment_dict


def sentiment_scores(df):
    scores = pd.Series(0.0, index=df.index)
    print ("Calculate sentiment scores...")
    for i in df.index:
        row = df.loc[i]["Title"] + ". " + df.loc[i]["Abstract"] + " " + df.loc[i]["Lesson(s) Learned"]
        sentiment_dict = calc_scores(row)
        scores.loc[i] = sentiment_dict['compound']
    print ("Complete.")
    return (-scores + 1) / 2