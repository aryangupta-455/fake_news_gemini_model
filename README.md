# Fake News Detection using BERT and Gemini LLM

## 📌 Project Overview

This project focuses on detecting fake news using two different approaches: a fine-tuned BERT model and the Gemini Large Language Model (LLM). The objective of the project is to compare traditional supervised transformer-based classification with modern zero-shot LLM reasoning and explanation generation.

The system classifies news articles as **Fake** or **Real** and also analyzes the explainability capabilities of LLM-based systems.

---

# 🚀 Features

* Fake news classification using a fine-tuned BERT model
* Zero-shot fake news detection using Gemini LLM
* Comparative analysis between both approaches
* Text-based explanation generation using Gemini
* User-friendly interface using Gradio
* Performance evaluation using standard ML metrics

---

# 🧠 Models Used

## 1. BERT (Fine-Tuned Model)

* Trained on labeled fake and real news datasets
* Performs supervised binary classification
* Outputs:

  * Fake News
  * Real News
* Evaluated using:

  * Accuracy
  * Precision
  * Recall
  * F1-score

### Working

The dataset is preprocessed and tokenized before being passed to the BERT model. The model learns patterns from labeled examples and predicts whether the input news article is fake or real.

---

## 2. Gemini LLM (Zero-Shot Approach)

* Uses prompt-based classification
* No dataset training required
* Generates:

  * Fake/Real prediction
  * Textual explanation

### Working

The input news text is directly sent to the Gemini API through prompts. Gemini analyzes the content using pre-trained language understanding capabilities and returns a classification along with reasoning.

---

# 📊 Dataset Information

* Source: Kaggle Fake News Dataset
* Total Records: 42,000

  * 21,000 Real News
  * 21,000 Fake News

### Features Used

* title
* text
* label / target

---

# ⚙️ Data Preprocessing

* Text cleaning
* Removal of extra spaces
* Label encoding
* Dataset splitting:

  * 70% Training
  * 15% Validation
  * 15% Testing

---

# 🖥️ Technologies Used

* Python
* BERT
* Gemini API
* Gradio
* Pandas
* Scikit-learn
* Matplotlib

---

# 📈 Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report

---
# Workflow of the Comparison System

```text
                    FAKE NEWS DETECTION PROJECT


┌──────────────────────────────┐
│      Approach 1 : BERT       │
└──────────────┬───────────────┘
               │
               ▼
      Dataset Preprocessing
               │
               ▼
       Fine-Tuning BERT Model
               │
               ▼
        Fake / Real Prediction
               │
               ▼
         Performance Evaluation
   (Accuracy, Precision, Recall,
             F1-score)



┌──────────────────────────────┐
│    Approach 2 : Gemini LLM   │
└──────────────┬───────────────┘
               │
               ▼
        User News Text Input
               │
               ▼
       Zero-Shot Prompting
               │
               ▼
      Gemini LLM Classification
               │
               ▼
      Fake / Real + Explanation
               │
               ▼
         Performance Evaluation
   (Accuracy, Precision, Recall,
             F1-score)



┌──────────────────────────────┐
│      Comparative Analysis    │
└──────────────────────────────┘

• BERT vs Gemini Results
• Accuracy vs Explainability
• Performance Trade-offs
```

# 🎯 Project Objective

The main objective of this project is to analyze the trade-off between:

* Accuracy of supervised transformer models
* Explainability and reasoning capabilities of LLMs

---

# 📌 Conclusion

The project demonstrates that fine-tuned transformer models like BERT provide strong classification accuracy, while LLM-based approaches such as Gemini offer better interpretability through natural language explanations. Both approaches have unique advantages depending on the application requirements.
