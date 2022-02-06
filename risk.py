from lbl2vec import Lbl2Vec
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument
from gensim.parsing.preprocessing import strip_tags


LBL2VEC_MODEL_PATH = "model/lbl2vec_model"


def tokenize(doc):
    # Input: document text string;
    # returns: tokenised document;
    # strip_tags: removes meta tags, simple_preprocess: converts into a list of lowercase tokens, removes numerical values and punktuation characters. 
    return simple_preprocess(strip_tags(doc), deacc=True, min_len=2, max_len=15)


def most_similar_risk(df, risk_labels, mode="train"):
    
    labels = pd.Series(risk_labels)
    
    # Tokenise and tag documents combined for Lbl2Vec training.
    tagged_docs = df.apply(lambda row: TaggedDocument(tokenize(row["Title"] + ". " + row["Abstract"] + " " + row["Lesson(s) Learned"] + " " + row["Driving Event"]), [str(row.name)]), axis=1)

    # initialise and train, or load the model.
    if mode=="train":
        print ("Training document classification model...")
        lbl2vec_model = Lbl2Vec(keywords_list=list(labels), tagged_documents=tagged_docs, label_names=list(labels.index), similarity_threshold=0.20, min_num_docs=1, epochs=30)
        lbl2vec_model.fit()
        lbl2vec_model.save(LBL2VEC_MODEL_PATH)
        print ("Complete.")
    else:
        print ("Loading document classification model...")
        try:
            lbl2vec_model = Lbl2Vec.load(LBL2VEC_MODEL_PATH)
        except:
            print ("Model loading error.")
            return
        print ("Complete.")
    
    # Compute similarity scores of learned document vectors, the scores are cosine similarities in [-1,1].
    model_docs_lbl_similarities = lbl2vec_model.predict_model_docs()
    model_docs_lbl_similarities.index = tagged_docs.index

    return model_docs_lbl_similarities["most_similar_label"]


def recommendation_score(df):
    recs = df["Recommendation(s)"]
    scores = pd.Series([len(x) for x in recs], index=recs.index)
    lims = [30, scores.quantile(0.95)]
    scores_ = (scores - lims[0]) / (lims[1] - lims[0]) 
    scores_[scores_ > 1] = 1
    scores_[scores_ < 0] = -1
    return scores_

    
def discretise(x, thresholds):
    if x < thresholds[0]:
        return "green"
    if x > thresholds[1]:
        return "red"
    return "yellow"
    
    
        
        