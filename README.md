# A Comparative Study of Machine Learning Methods

**Author:** Krishna Thakar  
**Faculty Advisors:** Dr. Mohamed Sheha and Dr. Emmanuel Thompson

## 📌 Objective

The goal of this project is to extract meaningful insights from Yelp restaurant reviews using Natural Language Processing (NLP). The analysis focuses on classifying reviews into **positive**, **neutral**, and **negative** categories

---

## 📊 Dataset

- **Source:** [Yelp Open Dataset](https://www.yelp.com/dataset)
- **Volume:** ~20,000 reviews (subset of original 7M)
- **Timeframe:** 2005–2022
- **Format:** Structured review text data (CSV/JSON)

---

## 🧹 Preprocessing Steps

- **Text Cleaning:** Lowercasing, punctuation & special character removal, stopword removal
- **Contraction Handling:** e.g., "don’t" → "do not"
- **Tokenization:** Splitting text into words
- **Lemmatization/Stemming**
- **Negation Handling:** e.g., "not good" → "not_good"
- **Vectorization:** TF-IDF, Word Embeddings (Word2Vec/BERT)
- **Data Balancing** Balancing the positive/negative/neutral reviews into 33% each.

---

## 🧠 Modeling Techniques

### Traditional ML Models
- Logistic Regression
- Support Vector Machine (SVM)
- Naïve Bayes
- Random Forest

### Deep Learning Models
- BiLSTM
- LSTM
- CNN
- RNN
- GRU

### Transformer-Based
- RoBERTa (fine-tuned)

---

## 📈 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Curve

---

## 🏆 Best Model Performance

### 🔢 Statistical Machine Learning Results (Table 1)
| Model               | Accuracy | Precision | Recall   | F1 Score | AUC     |
|--------------------|----------|-----------|----------|----------|---------|
| **SVM**            | **0.76103** | **0.76322**   | **0.76103**  | **0.76172**  | **0.90915** |
| Logistic Regression| 0.76041  | 0.76448   | 0.76411  | 0.76013  | 0.90757 |
| Naïve Bayes        | 0.72671  | 0.73246   | 0.72671  | 0.72860  | 0.87732 |
| Random Forest      | 0.71751  | 0.71593   | 0.71751  | 0.71592  | 0.87531 |

📌 *SVM* achieved the best performance among traditional models, with a strong AUC of **0.90915**.

---

### 🤖 Deep Learning & Transformer-Based Results (Table 2)
| Model      | Accuracy | Precision | Recall   | F1 Score | AUC     |
|------------|----------|-----------|----------|----------|---------|
| **RoBERTa**| **0.80112** | **0.80502** | **0.80112** | **0.80253** | **0.93237** |
| CNN        | 0.74746  | 0.74909   | 0.74746  | 0.74817  | 0.89802 |
| BiLSTM     | 0.73280  | 0.73371   | 0.73280  | 0.73323  | 0.88507 |
| LSTM       | 0.33614  | 0.58349   | 0.33614  | 0.17434  | 0.50671 |
| RNN        | 0.33614  | 0.58349   | 0.33614  | 0.17434  | 0.50671 |
| GRU        | 0.34144  | 0.41822   | 0.34144  | 0.19040  | 0.50927 |

🚀 **RoBERTa** significantly outperformed all other models with:
- **Highest Accuracy**: 0.80112
- **Best AUC**: 0.93237
- **Strong F1 Score**: 0.80253

✅ **RoBERTa is the overall best-performing model in this study**, showing the effectiveness of transformer-based architectures in sentiment classification.

---

## 🧠 Key Findings

- **RoBERTa outperformed** traditional and deep learning models in accuracy and F1-score.
- Transformer models effectively captured contextual sentiment.


---

## 🧩 Tools & Libraries
- Python (Pandas, NumPy, Scikit-learn)
- NLTK / SpaCy
- TensorFlow / PyTorch
- HuggingFace Transformers
- Matplotlib / Seaborn / Plotly

---

## 📍 Future Work
- Expand aspect-based sentiment analysis (ABSA)
- Use real-time sentiment tracking for businesses
- Deploy model as a REST API or Streamlit dashboard

---

## 📬 Contact
**Krishna Thakar**  
📧 krishna161003@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/krsnathkr/) | [GitHub](https://github.com/krsnathkr)  

---