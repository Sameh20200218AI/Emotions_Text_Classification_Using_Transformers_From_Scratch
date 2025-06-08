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

✅ Preprocessing Steps
Loaded the dataset using Pandas.

Used a pre-trained NLP model to clean and tokenize text:

Normalized text

Removed emojis, tags, stop words, and special characters

Preserved negation words

Created a new column for cleaned text.

📊 Data Visualization
Histogram of sequence lengths (original vs preprocessed text)

WordClouds for each emotion class

Bar and Pie Charts showing class distribution

🧪 Train-Test Split
Train: 90%

Test: 10%

Due to the large dataset size, this split is efficient.

All samples padded to the max sequence length (69 tokens)

text
Copy
Edit
Training shape: (375128, 69)  
Testing shape: (41681, 69)
🧠 Transformer Model (From Scratch)
Implemented the following from scratch:

TokenAndPositionEmbedding class

TransformerEncoder class

🔧 Model Architecture
embed_dim: 100

num_heads: 8

feedforward_dim: 64

Early stopping: Monitor val_loss, stop after 3 non-improving epochs

Best model weights saved

🏋️ Training Parameters
Epochs: 20

Batch size: 1024

Validation split: 15%

Training stopped at Epoch 5

text
Copy
Edit
Training Accuracy: 91.89%  
Validation Accuracy: 89.16%
📈 Evaluation
Test Accuracy: 90.36%

Test Loss: 0.2006

Printed classification report and plotted confusion matrix

Tested custom input examples – model performed very well.

Despite class imbalance, the model achieved:

~91% Accuracy

Strong F1 Scores

🚀 Model Deployment
Used Gradio to build an interactive UI for emotion classification.

Features:
Load and preprocess user input

Predict emotion class with associated emoji 😊😢😡 etc.

Display confidence scores for all classes

💾 Model Saved
Final trained model is saved and reusable for deployment or further training.

📌 Conclusion
This project successfully built a powerful and interpretable Transformer-based Emotion Classifier from scratch, delivering excellent results on a large and imbalanced dataset. The interactive Gradio app adds real-world usability for end users.

🔗 Future Improvements
Handle multi-label emotions

Expand to multilingual datasets

Fine-tune pretrained Transformers like BERT for comparison

👨‍💻 Author
Sameh Raouf
