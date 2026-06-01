# 🌐 Multilingual Emotion Detection of Tamil and Telugu E-Commerce Product Reviews

> **Course:** SWE1017 – Natural Language Processing  
> **Institution:** School of Computer Science, Engineering and Information Systems (SCORE)  
> **Guide:** Dr. Senthil Kumar M

---

## 👥 Team Members

| Name | Register No. |
|------|--------------|
| Pavithra S | 22MIS0180 |
| Motakatla Kavya Sree | 22MIS0456 |

---

## 📌 Overview

This project develops a **multilingual emotion detection system** for Tamil and Telugu e-commerce product reviews. Unlike traditional sentiment analysis (positive/negative/neutral), the system classifies customer reviews into **8 distinct emotion categories**, providing richer, more actionable insights for businesses.

### 🎭 Emotion Categories
`Happy` · `Sad` · `Confused` · `Angry` · `Surprised` · `Disgusted` · `Neutral` · `Disappointed`

---

## 🏗️ Architecture

```
Raw Reviews (Amazon / Flipkart / Snapdeal)
        ↓
   Data Collection (Octoparse Web Scraping)
        ↓
   Preprocessing (Unicode Normalization, Stopword Removal, Tokenization via IndicNLP)
        ↓
   Zero-Shot Labeling (xlm-roberta-large-xnli)
        ↓
   Dataset Balancing (per-class equalization)
        ↓
   ┌────────────────────────────────┐
   │   Model Training               │
   │  ├── Logistic Regression (TF-IDF) │
   │  ├── Random Forest            │
   │  ├── XGBoost                  │
   │  └── XLM-RoBERTa-base (fine-tuned) │
   └────────────────────────────────┘
        ↓
   Evaluation (Accuracy, F1, Confusion Matrix, SHAP/LIME)
        ↓
   Streamlit Web Application
```

---

## 📊 Results

| Model | Validation Accuracy | Weighted F1 |
|-------|-------------------|-------------|
| Logistic Regression (TF-IDF) | ~80% | ~0.80 |
| Random Forest | ~80% | ~0.80 |
| XGBoost | ~80% | ~0.80 |
| **XLM-RoBERTa-base (fine-tuned)** | **96.8%** | **~0.97** |

### Per-Class F1 Scores (XLM-RoBERTa)
| Emotion | F1-Score |
|---------|----------|
| Angry | ~0.99 |
| Disgusted | 1.00 |
| Happy | ~0.98 |
| Neutral | ~0.98 |
| Sad | ~0.97 |
| Confused | ~0.97 |
| Disappointed | ~0.92 |
| Surprised | ~0.84 *(hardest class)* |

> **Final Test Set Accuracy: 98%**

---

## 🗂️ Project Structure

```
├── NLP_project.ipynb
├── NLP_DATASET.csv                  # Annotated Tamil & Telugu review dataset
├── balanced_dataset_light.csv       # Class-balanced dataset
├── app.py                           # Streamlit web application
├── mbert_label_encoder.pkl          # Saved LabelEncoder
├── my_champion_model_96_8/          # Saved XLM-RoBERTa model weights
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for transformer training)

### Install Dependencies

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn
pip install streamlit neattext indic-nlp-library langdetect xgboost shap lime
pip install ktrain datasets accelerate
```

---

## 🚀 Running the App

```bash
streamlit run app.py
```

Make sure the model directory and label encoder are at the correct paths before launching:
- Model: `./my_champion_model_96_8/`
- Encoder: `./mbert_label_encoder.pkl`

---

## 🔬 Methodology

### 1. Data Collection
- Sources: Amazon India, Flipkart, Snapdeal, regional Tamil Nadu e-commerce platforms
- Tools: **Octoparse** (no-code web scraping), **Google Translate API** (translation fallback)

### 2. Preprocessing
- Unicode normalization for Tamil & Telugu scripts
- Stopword removal via `neattext`
- Language-specific tokenization via `indic_nlp_library`
- Language validation using `langdetect`

### 3. Data Labeling
- Zero-shot classification using `joeddav/xlm-roberta-large-xnli`
- Manual validation and annotation refinement
- Dataset balancing (~85 samples per class)

### 4. Feature Extraction
- TF-IDF (unigrams + bigrams) for classical ML models
- Contextual embeddings via XLM-RoBERTa tokenizer for deep learning

### 5. Model Training
- **Classical ML:** Logistic Regression, Random Forest (Bagging), XGBoost (Boosting)
- **Transformer:** `xlm-roberta-base` fine-tuned via Hugging Face `Trainer` API
  - 5 epochs, learning rate `2e-5`, weight decay `0.01`, `fp16=True`

### 6. Evaluation
- Accuracy, Precision, Recall, F1-Score (weighted)
- Confusion matrix heatmaps
- SHAP & LIME for model explainability

---

## 📈 Key Findings

- **Transformer models significantly outperform classical ML** (96.8% vs ~80%) for multilingual emotion classification.
- **Dataset balancing is critical** — the raw dataset was highly skewed toward "Happy", causing bias.
- **"Surprised" is the hardest emotion** to classify, frequently confused with "Sad" and "Disappointed".
- **mBERT/XLM-RoBERTa handle code-mixed text** (Tamil/Telugu + English) effectively without explicit handling.

---

## 🔮 Future Work

- Fine-tune on domain-specific Tamil/Telugu emotion datasets
- Expand to Hindi, Kannada, and Malayalam
- Deploy on AWS SageMaker / Hugging Face Spaces for scalable inference
- Build interactive Plotly/Power BI emotion dashboards
- Hybrid ensemble combining transformer + classical model outputs
- Integrate with e-commerce recommendation systems
- Mobile app deployment using TensorFlow Lite

---

## 📚 References

1. Abdullah & Rusli. "Multilingual Sentiment Analysis: A Systematic Literature Review." *Pertanika Journal*, 2021.
2. Boiy & Moens. "A machine learning approach to sentiment analysis in multilingual web texts." *Information Retrieval*, 2009.
3. Shah & Kaushik. "Sentiment analysis on Indian indigenous languages." *arXiv:1911.12848*, 2019.
4. Aggarwal et al. "IndicXNLI: Evaluating multilingual inference for Indian languages." *arXiv:2204.08776*, 2022.
5. Kumar et al. "Emotion recognition in Hindi text using multilingual BERT transformer." *Multimedia Tools and Applications*, 2023.
6. Deshpande et al. "When is BERT multilingual?" *arXiv:2110.14782*, 2021.
7. Kalaivani & Thenmozhi. "Multilingual sentiment analysis in Tamil, Malayalam, and Kannada code-mixed posts using MBERT." *FIRE*, 2021.
8. Libovický et al. "How language-neutral is multilingual BERT?" *arXiv:1911.03310*, 2019.
9. Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers." *NAACL-HLT*, 2019.

---

## 📄 License

This project was developed for academic purposes under VIT University's NLP course (SWE1017).
