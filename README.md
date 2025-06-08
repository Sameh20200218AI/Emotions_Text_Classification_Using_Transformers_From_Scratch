
# 🎭 Emotion Text Classification using Transformers (Built from Scratch)


This project focuses on building a robust **Emotion Text Classifier** using a **Transformer architecture implemented entirely from scratch**. The goal is to classify input text into one of six emotional categories:

> `sadness`, `joy`, `love`, `anger`, `fear`, and `surprise`

---

## 📚 Libraries & Tools Used

- **TensorFlow**, **Keras** – Deep Learning model construction  
- **scikit-learn** – Data preprocessing, train/test splitting, evaluation metrics  
- **Pandas**, **NumPy** – Data manipulation and analysis  
- **Seaborn**, **Matplotlib** – Visualizations  
- **WordCloud** – Generating word clouds for each emotion  
- **spaCy** – Text cleaning, tokenization, and lemmatization  
- **emoji** – Emoji detection and removal  
- **Gradio** – Web-based GUI for model interaction and deployment  

---

## 📑 Dataset

- **Name:** Emotions Dataset  
- **Samples:** 416,809  
- **Classes:** 6  
  ```python
  class_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
  ```

---

## 🧼 Text Preprocessing

- Loaded using **Pandas**
- Text cleaned and tokenized using a pre-trained **spaCy** model
- Preprocessing pipeline includes:
  - Text normalization (lemmatization)
  - Removing:
    - Emojis
    - HTML tags
    - Stop words
    - Special characters  
  - Preserving negation words
- Added a new column with the cleaned version of the text

---

## 📊 Data Visualization

- **Histogram:** Distribution of sequence lengths comparing original and cleaned texts, illustrating the impact of preprocessing on text length.

<div align="center">
  <img src="https://github.com/Sameh20200218AI/Emotions_Text_Classification_Using_Transformers_From_Scratch/blob/main/Sequence_Length_Distribution.png" alt="Histogram of Sequence Lengths" width="800"/>
</div>

---
- **Word Clouds:** Visual representations of high-frequency words for each emotion class, highlighting key terms associated with each emotion.

<div align="center">
  <img src="https://github.com/Sameh20200218AI/Emotions_Text_Classification_Using_Transformers_From_Scratch/blob/main/Word_Cloude_All_Classes.png" alt="Word Cloud Visualization" width="800"/>
</div>

---

- **Bar Chart** and **Pie Chart:** Visualize the distribution of samples across the six emotion classes.  
> Note: The dataset is imbalanced, with `joy` and `sadness` classes dominating the sample count.

<div align="center">
  <img src="https://github.com/Sameh20200218AI/Emotions_Text_Classification_Using_Transformers_From_Scratch/blob/main/Count_Number_of_Samples.png" alt="Bar Chart of Class Distribution" width="800"/>
</div>

---


## 🧪 Dataset Split & Padding

- **Training Set:** 90%  
- **Testing Set:** 10%  
  > Due to large sample size, this split was efficient and optimal.

- All text samples were padded to match the **maximum sequence length (69 tokens)**

```text
Training shape: (375,128, 69)  
Testing shape: (41,681, 69)
```

---

## 🧠 Transformer Architecture (Implemented from Scratch)

### ✅ Components:
- `TokenAndPositionEmbedding` class  
- `TransformerEncoder` class

### 🔧 Model Architecture:

| Parameter        | Value      |
|------------------|------------|
| `embed_dim`      | 100        |
| `num_heads`      | 8          |
| `feedforward_dim`| 64         |

- Early stopping used on `val_loss` with patience of 3 epochs  
- Best weights saved during training  

---

## 🏋️ Training Configuration

| Setting              | Value   |
|----------------------|---------|
| Epochs               | 20      |
| Batch Size           | 1024    |
| Validation Split     | 15%     |
| Early Stopping       | ✅      |
| Stopped At           | Epoch 5 |

```text
Training Accuracy: 91.89%  
Validation Accuracy: 89.16%
```
---
<div align="center">
  <img src="https://github.com/Sameh20200218AI/Emotions_Text_Classification_Using_Transformers_From_Scratch/blob/main/Training_History.png" alt="Histogram of Sequence Lengths" width="800"/>
</div>
---

## 📈 Evaluation Results

| Metric         | Value     |
|----------------|-----------|
| Test Accuracy  | 90.36%    |
| Test Loss      | 0.2006    |

- Generated a **Classification Report**  
- Plotted **Confusion Matrix** using Seaborn heatmap
  <div align="center">
  <img src="https://github.com/Sameh20200218AI/Emotions_Text_Classification_Using_Transformers_From_Scratch/blob/main/Confusion_Matrix.png" alt="Histogram of Sequence Lengths" width="500"/>
</div>

- Evaluated with **custom input examples**  
- Model generalizes well and handles unseen text confidently

✅ Despite class imbalance:
- Achieved ~91% accuracy  
- Strong F1-score across all categories  

---

## 🚀 Deployment using Gradio

An interactive UI was built with **Gradio** for real-time emotion prediction.

### Features:
- Clean and preprocess user input
- Predict emotion category with emoji output 😊😢😡😨
- Display confidence scores for **all 6 classes**

> Model is fully functional and reusable for production or further fine-tuning.

---

## 💾 Model Persistence

- Final trained model and tokenizer are **saved locally**
- Can be reloaded for further training, batch inference, or deployment  

---

## 📌 Conclusion

This project demonstrates how a **Transformer-based Emotion Classifier** can be built from the ground up and deliver **excellent results** on a large, imbalanced dataset. With preprocessing, visualization, training, evaluation, and deployment all integrated, this project stands as a **complete end-to-end NLP system**.

---

## 🚧 Future Improvements

- Extend to **multi-label emotion detection**
- Apply to **multilingual** emotion classification datasets
- Compare with fine-tuned **pretrained Transformers** (e.g. BERT, RoBERTa)
- Perform **ablation studies** on architecture components

---

## 👨‍💻 Author

**Sameh Raouf**  
🎓 AI Engineer  
---

