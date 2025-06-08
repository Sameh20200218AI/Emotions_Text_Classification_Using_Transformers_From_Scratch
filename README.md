# Emotion Text Classification using Transformers (Built from Scratch)

This project focuses on building a robust **Emotion Text Classifier** using a Transformer model implemented **from scratch**. The goal is to classify text data into one of six emotional categories: `sadness`, `joy`, `love`, `anger`, `fear`, and `surprise`.

---

## 📚 Libraries Used

- `TensorFlow`, `Keras` – Deep Learning and model building  
- `scikit-learn` – Data preprocessing and evaluation  
- `Pandas`, `NumPy` – Data handling  
- `Seaborn`, `Matplotlib` – Visualization  
- `WordCloud` – Word cloud visualization  
- `spaCy` – Text normalization and tokenization  
- `emoji` – Emoji handling  
- `Gradio` – Model deployment with GUI

---

## 📑 Dataset

- **Name:** Emotions Dataset  
- **Samples:** 416,809  
- **Classes:** 6  
  ```python
  class_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
