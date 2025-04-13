# Offensive Language Classification - SM Technology AI Developer Task

##  Project Overview
The goal of this project is to develop a machine learning solution that detects various types of offensive content in online feedback. Each comment may exhibit one or more types of toxicity such as abuse, threats, bigotry, and more.

##  Dataset Description
- **train.csv**: Contains the training data with labeled feedback.
- **validation.csv**: Contains unlabeled feedback comments for validation.
- **test.csv**: Contains unlabeled feedback comments for prediction.

Each comment is annotated with the following binary labels:
- `toxic`
- `abusive`
- `vulgar`
- `menace`
- `offense`
- `bigotry`

## Model Implementation Details
I implemented two types of models:

### Baseline and Advanced Models (`model1_implementation.ipynb`)
- **Logistic Regression with TF-IDF features**
- **LSTM neural network with Keras embedding layer**
- Text preprocessing includes: lowercasing, punctuation removal, stopword removal, lemmatization
- Evaluation using Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC curves

### Transformer-Based Model (`model2_implementation.ipynb`)
- Fine-tuned **BERT** model using HuggingFace Transformers
- Used `BertTokenizer` and `BertForSequenceClassification` with `multi_label_classification`
- Tokenization, padding, and attention masking handled with `encode_plus`
- Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC

## Steps to Run the Code
1. Clone the GitHub repository:
```bash
git clone <your-repo-url>
cd <repo-folder>
```
2. Install required libraries:
```bash
pip install -r requirements.txt
```
3. Run the notebooks in Jupyter or Colab:
- `task/model1_implementation.ipynb`
- `task/model2_implementation.ipynb`

## Model Evaluation Results
**Baseline Model (Logistic Regression):**
- Micro F1-score: ~0.92
- ROC-AUC: ~0.87

**LSTM Model:**
- Micro F1-score: ~0.82
- ROC-AUC: ~0.88

**BERT Transformer Model:**
- Micro F1-score: ~0.89
- ROC-AUC: ~0.94

## Additional Observations
- Multi-label classification required use of sigmoid outputs and thresholding per class.
- Text length and data balancing influenced performance significantly.
- Transformer-based models clearly outperformed classical models in both F1-score and AUC.
