# NASA Project Risk Prediction AI Model       


<!-- ABOUT THE PROJECT -->
## About The Project

The given model predicts the future project risks based on the historical lesson(s) learned data. In the project, the following unsupervised deep learning algorithms are trained: (1) text embedding and text similarity (Doc2Vec), (2) text classification (Lbl2Vec), (3) sentiment analysis. The results are converted to the NASA standard cost/schedule/technical/programmatic – green/yellow/red system.    

<!-- GETTING STARTED -->
## Getting Started

The code is written in Python 3.9. All the input data and pretrained models are places in the folders `/data` and `/model` respectively. 

### Prerequisites

The following standard Python libraries are required to run the project.

* os
* pandas
* json
* re
* matplotlib

### Installation

1. Install [Word2Vec](https://radimrehurek.com/gensim/models/doc2vec.html):
```
pip install gensim
```

2. Install [Lbl2Vec](https://pythonrepo.com/repo/sebischair-Lbl2Vec):
```
pip install lbl2vec
```

3. Install [VADER (Valence Aware Dictionary and sEntiment Reasoner)](https://pypi.org/project/vaderSentiment/):
```
pip install vaderSentiment
```

4. Install Python bindings for the PDF toolkit and renderer MuPDF:
```
pip install PyMuPDF
```

<!-- USAGE EXAMPLES -->
## Usage

### Run

To run the code in the inference mode (the pretrained models will be loaded), type:
```
python main.py
```

The output will be the NASA standard risk calssification for one of test projects (specified in the `config` dictionary).

### Configuration

The `config` dictionary (in `main.py`) defines the parameters of the model: 

* `lessons_learned_file` – a path to the lesson(s) learned CSV table;
* `project_plan_file` – a path to the project plan PDF;
* `project_data_file` – a path to the extracted project plan text JSON (used either `project_plan_file` or `project_data_file`);
* `parse_project_plan` – if `True`, the project plan PDF will be parsed, otherwise, the extracted JSON will be read;
* `train` – if `True`, all the models will be trained from scratch, otherwise, the code will operate in the inference mode;
* `recommendation_weight` – a weight of normalised text length in overall risk calculation (I recommend to keep it low);
* `num_similar_docs` – number of similar matches of the lesson(s) learned entries with the target project plan data (5 is optimal);
* `risk_thresholds` – threshold values of green/yellow and yellow/red risk levels. 

### Risk labels

The risk keywords `risk_labels` in `main.py` contains characteristic keyworks taken from the standard [NASA risk classification](https://www.nasa.gov/seh/6-4-technical-risk-management), but can also be adjusted.   


<!-- RFESULTS -->
## Results

### Astrobee predicted project risk 

[Astrobee]: images/screenshot.png

[Synbio]: images/synbio.png

<!-- LICENSE -->
## License

Distributed under the TBAL (To Be Added Later) License. 

<!-- CONTACT -->
## Contact

Alexander Poplavsky alexanderpoplavsky@yahoo.com
