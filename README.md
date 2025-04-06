# Medical Text Classification with BioMedBERT


## Overview
This project implements a medical text classification system using BioMedBERT, a domain-specific BERT model pre-trained on biomedical literature. The solution includes complete pipeline from data preprocessing to model evaluation, comparing traditional TF-IDF approaches with modern transformer-based architectures.

Key Features:
- Text cleaning and preprocessing pipeline
- TF-IDF + Logistic Regression baseline
- PubMedBERT fine-tuning implementation
- Hyperparameter tuning and evaluation
- Class imbalance handling
- GPU-accelerated training

## Dataset
**COVID-19 Research Papers** (`covid_df.csv`)
- 222 annotated medical abstracts related to COVID-19 research
- Columns:
  - `doc_id`: Unique document identifier
  - `Title`: Paper title
  - `Abstract`: Full text abstract
  - `Year`: Publication year (2020-2023)
  - `Split`: Train/Valid/Test split
  
Dataset Statistics:
- 80% Training (177 samples)
- 10% Validation (22 samples)
- 10% Testing (23 samples)

## Text Preprocessing
1. **spaCy Processing**:
   - Stopword removal
   - Lemmatization
   - Non-alphabetic character removal
   - Whitespace normalization

2. Special Handling:
   - Custom cleaning function applied to abstracts
   - TF-IDF vectorization with feature selection
   - Train/test separation to prevent data leakage

## Approaches Used

### 1. Traditional ML Approach (TF-IDF + Logistic Regression)
- `TfidfVectorizer` with max feature tuning
- Logistic Regression classifier
- Hyperparameter search for optimal vocabulary size
 - Example hyperparameter tuning code
    max_features_list = [50, 100, 500, 1000]
    find_best_max_features(df, max_features_list)
- F1-score evaluation

### 2. BioMedBERT (Transformers)
- Hugging Face `transformers` library integration
- BioMedBERT tokenization and embedding
- GPU-accelerated training
- Fine-tuning strategies:
  - Learning rate optimization
  - Batch size tuning
  - Early stopping
   -BioMedBERT Fine-tuning
       AutoModelForSequenceClassification.from_pretrained("microsoft/BiomedBERT")


### Key Comparisons
- Traditional vs. deep learning approaches
- Impact of text preprocessing on model performance
- Class-wise evaluation metrics


## Evaluation
Metrics:
- Macro/weighted F1-score
- Confusion matrices

Key Results:
- TF-IDF + Logistic Regression achieved 0.63 F1-score
- PubmedBert architecture trained from scratch without pretrained weights
- PubMedBERT with pretrained weights,fine tuning improved performance to 0.72 F1
- Optimal TF-IDF vocabulary size: 100 features


## Installation

### Google Colab
[![Open In Colab](https://colab.research.google.com/github/sarim711/BiomedBERT_Medical-Text_Classification/blob/main/BioMedBERT_MedicalText_Classification.ipynb)  

No setup needed - just run all notebook cells after uploading dataset

### Local Execution
```bash
# Clone repository
git clone https://github.com/sarim711/BiomedBERT_Medical-Text_Classification.git
cd BiomedBERT_Medical-Text_Classification

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download spacy model
python -m spacy download en_core_web_sm
```

## How to Use

1. Place `covid_df.csv` in `/data` directory
2. Launch Jupyter:
```bash
jupyter notebook BioMedBERT_MedicalText_Classification.ipynb
```
3. Run cells sequentially through all three approaches


