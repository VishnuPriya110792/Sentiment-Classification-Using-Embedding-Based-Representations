REPORT

Title:-

Sentiment Classification Using Embedding-Based Representations

1. Introduction

With the rapid growth of social media platforms, understanding public sentiment
has become essential for organizations to respond to user opinions in real time.
This project aims to build an automated sentiment classification system capable
of categorizing tweets into positive, negative, or neutral classes using semantic
embeddings.

2. Dataset Description

The dataset used for this project is sourced from Kaggle and consists of
approximately 27,000 tweets. Each tweet is labeled with one of three sentiment
classes: Positive, Negative, or Neutral. Only the text and sentiment columns
were used for modeling.

3. Methodology

Text preprocessing involved removing URLs, mentions, special characters,
and standardizing text casing. Sentence-level embeddings were generated using
a pre-trained Sentence Transformer model, which maps each tweet to a dense
numerical vector capturing semantic meaning.

These embeddings were used as input features for two classification models:
Logistic Regression and XGBoost. The dataset was split into training and testing
sets to evaluate generalization performance.

4. Evaluation Metrics

Model performance was evaluated using accuracy, precision, recall, and F1-score.
A confusion matrix was used to analyze class-wise prediction behavior and
misclassification patterns.

5. Results and Analysis

XGBoost outperformed Logistic Regression, particularly in distinguishing
positive and negative sentiments. Neutral sentiment proved to be the most
challenging class due to overlapping semantic cues. Error analysis revealed
that sarcasm and mixed-emotion tweets were frequent sources of misclassification.

6. Conclusion

The project demonstrates that embedding-based representations significantly
improve sentiment classification for short-form text compared to traditional
methods. While results are strong, further improvements are possible through
fine-tuned transformer models and sarcasm-aware approaches.

7. Future Scope

Future extensions include multilingual sentiment analysis, domain adaptation,
and deployment as a real-time sentiment analysis application.
