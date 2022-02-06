import pandas as pd
from gensim.parsing.preprocessing import preprocess_documents, preprocess_string
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


DOC2VEC_MODEL_PATH = "model/doc2vec_model"


def similarity(df, text, mode="train"):
    text_corpus = (df["Abstract"] + " " + df["Lesson(s) Learned"]).values
    processed_corpus = preprocess_documents(text_corpus)
    tagged_corpus = [TaggedDocument(d, [i]) for i, d in enumerate(processed_corpus)]
    
    if mode=="train":
        print ("Training document embedding model...")
        doc2vec_model = Doc2Vec(tagged_corpus, dm=0, vector_size=200, window=2, min_count=1, epochs=100, hs=1)
        doc2vec_model.save(DOC2VEC_MODEL_PATH)
        print ("Complete.")
    else:
        print ("Loading document embedding model...")
        try:
            doc2vec_model = Doc2Vec.load(DOC2VEC_MODEL_PATH)
        except:
            print ("Model loading error.")
            return
        print ("Complete.")
    
    new_doc = preprocess_string(text)
    test_doc_vector = doc2vec_model.infer_vector(new_doc)
    sims = doc2vec_model.dv.most_similar(positive = [test_doc_vector])
    return pd.Series([x[1] for x in sims], index=[x[0] for x in sims])
