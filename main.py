import os
import pandas as pd

from data import load_lessons_learned_data, load_project_data
from extractor import extract_project_data
from sentiment import sentiment_scores
from risk import most_similar_risk, recommendation_score, discretise
from embedding import similarity

from visualise import draw_risk


config = {"lessons_learned_file": "lessons_learned.csv",
          "project_plan_file": "project_plan.pdf",
          "project_data_file": "astrobee.json",
          "parse_project_plan": False,
          "train": False,
          "recommendation_weight": 0.25,
          "num_similar_docs": 5,
          "risk_thresholds": [0.25, 0.60]} 


risk_labels =  {"cost": ["cost", "funding"],
           "schedule": ["schedule", "time", "plan"],
           "technical": ["technical", "design", "system", "failure", "damage"],
           "programmatic": ["contractor", "organization", "relationship", "government"]}


if __name__ == "__main__":
    
    lessons_learned_path = os.path.join("data", config["lessons_learned_file"])
    
    df = load_lessons_learned_data(lessons_learned_path)
    
    if config["parse_project_plan"]:
        project_plan_path = os.path.join("data", config["project_plan_file"])
        chapter_to_extract = "technicalapproach"
        project_data = extract_project_data(project_plan_path, chapter_to_extract)
    else:
        project_data_path = os.path.join("data", config["project_data_file"])
        project_data = load_project_data("data/astrobee.json")["technical_approach"]
        
    # Train or inference mode.
    mode = "inference"
    if config["train"]:
        mode = "train"
    print (mode.upper() + " mode.")
    
        
    # Calculate most similar risk.
    df["risk"] = most_similar_risk(df, risk_labels, mode)
    
    # Calculate sentiment-based scores.
    sent_scores = sentiment_scores(df)
    
    # Calculate recommendation-based scores.
    recom_scores = recommendation_score(df)
    
    # Combine scores.
    weights = pd.Series(config["recommendation_weight"], index=df.index)
    weights[recom_scores==-1] = 0
    scores = weights * recom_scores + (1 - weights) * sent_scores
    df["score"] = scores
    
    # Calculate document embedding.
    sims = similarity(df, project_data, mode)    
    index = []
    for i in sims.index:
        if i in df.index:
            index.append(i)
    sims = sims[index]
    sims = sims.nlargest(min(config["num_similar_docs"], len(sims)))
    
    # Extract sample of similar documents.
    sample = df.loc[sims.index, ["risk", "score"]]
    
    # Calculate risk.
    risk = {}
    for label in risk_labels:
        if label in sample["risk"].values:
            index = sample[sample["risk"]==label].index
            weights = sims[index] / sims[index].sum()
            risk[label] = (sample.loc[index, "score"] * weights).sum()
        else:
            risk[label] = 0
    
    # Discretise and visualise risk.
    discrete_risk = {key: discretise(risk[key], config["risk_thresholds"]) for key in risk}
    draw_risk(discrete_risk)

