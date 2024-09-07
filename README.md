# Text Generation Using LSTM and Bidirectional LSTM

This project focuses on generating text using deep learning models, specifically **Long Short-Term Memory (LSTM)** and **Bidirectional LSTM** networks. By training the model on a literary text, the goal is to predict the next word in a sequence, making the model capable of generating coherent and contextually relevant text. The dataset is extracted from the **Gutenberg Corpus**, a rich source of classical literature, and preprocessed using **NLTK**.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Model Pipeline](#model-pipeline)
4. [Detailed Explanation of Key Components](#detailed-explanation-of-key-components)
5. [Model Training and Limitations](#model-training-and-limitations)
6. [Usage Instructions](#usage-instructions)
7. [Example Predictions](#example-predictions)
8. [Requirements and Installation](#requirements-and-installation)
9. [License](#license)

---

## Project Overview

The core objective of this project is to train a machine learning model to predict the next word in a sentence or phrase. By leveraging **LSTM** and **Bidirectional LSTM**, the model learns complex sequence patterns and generates text based on the given input.

- **Dataset**: Extracted from William Blake’s "Poems" in the NLTK **Gutenberg Corpus**.
- **Objective**: To develop a deep learning model capable of generating contextually relevant words based on preceding words.

---

## Technologies Used

### Libraries & Frameworks:
1. **Pandas**: Though primarily used for data manipulation, here it provides efficient handling of dataset loading.
2. **NumPy**: Essential for numerical operations and efficient array handling, particularly when managing sequences and padding.
3. **Matplotlib**: (Optional) Provides tools for visualizing data, particularly useful for analyzing training performance.
4. **NLTK (Natural Language Toolkit)**: Used for accessing and processing text from the **Gutenberg Corpus**. This is a vital tool for working with linguistic data.
   - **Gutenberg Corpus**: A large collection of texts, including works from Shakespeare, Austen, and Blake, provided by the NLTK library.
5. **TensorFlow / Keras**: The deep learning framework used to implement the LSTM and Bidirectional LSTM models. TensorFlow provides robust tools for neural network modeling, while Keras simplifies model construction.
6. **Scikit-learn**: Provides utilities for data preparation, including the train-test split.

### Core Keras Components:
- **Embedding Layer**: Converts words into dense vectors (embeddings) based on their integer representations, facilitating machine learning on textual data.
- **LSTM (Long Short-Term Memory)**: A recurrent neural network (RNN) architecture used to capture long-term dependencies in sequential data.
- **Bidirectional LSTM**: Extends the capabilities of LSTM by processing data in both forward and backward directions, enhancing context understanding.
- **Dense Layer**: A fully connected neural network layer used to predict the next word in the sequence.
- **Dropout**: A regularization technique that prevents overfitting by randomly setting a fraction of the input units to zero during training.
- **BatchNormalization**: Normalizes the output of the previous layer, stabilizing and accelerating training by reducing the risk of gradient vanishing/exploding.
- **Adam Optimizer**: A variant of gradient descent that adapts learning rates for each parameter, ensuring efficient convergence.

---

## Model Pipeline

The project follows a systematic approach to preprocess data, build the model, train it, and make predictions. Below are the steps involved:

1. **Data Collection**:
   - **Gutenberg Corpus** is accessed using NLTK to extract literary texts.
   - In this case, the dataset is **William Blake's "Poems"**.
   - The raw text is saved locally for future reference.

2. **Text Preprocessing**:
   - **Lowercasing**: Converts all text to lowercase to maintain uniformity.
   - **Tokenization**: The text is split into individual words and each unique word is mapped to an integer (word index).
   - **Padding**: The sequences are padded to ensure all input sequences are of the same length for model training.

3. **Model Architecture**:
   - **Embedding Layer**: Converts input tokens (integers) into dense word vectors (embeddings), capturing semantic information.
   - **Bidirectional LSTM**: Processes the input sequence both forwards and backwards to improve context understanding.
   - **Dropout Layers**: Applied to prevent overfitting by deactivating random neurons during training.
   - **Batch Normalization**: Normalizes intermediate layer outputs, stabilizing the training process.
   - **Dense Layer**: Used for making predictions, with the number of output units equal to the vocabulary size, and the softmax activation function for multi-class classification.

4. **Training and Validation**:
   - **EarlyStopping**: Monitors the validation loss and stops training if the model stops improving, preventing overfitting.
   - **ReduceLROnPlateau**: Dynamically adjusts the learning rate when validation performance plateaus, ensuring effective learning even after performance stagnation.
   - **Train-Test Split**: The data is split into training (80%) and testing (20%) sets for model evaluation.

5. **Prediction**:
   - The trained model predicts the next word in a given text sequence by utilizing the learned patterns from the training phase.

---

## Detailed Explanation of Key Components

### 1. Tokenizer:
The **Tokenizer** from Keras is used to transform the text into sequences of integers where each integer represents a word in the text. This transformation is crucial for enabling the model to work with textual data.

### 2. Padding Sequences:
The sequences created from the text can vary in length, but neural networks require fixed-size inputs. The **pad_sequences** function ensures all input data has the same length by adding zeros at the beginning of shorter sequences.

### 3. Embedding Layer:
The **Embedding Layer** takes the tokenized sequences and transforms them into dense vectors where each word is represented by a vector of floating-point values. These embeddings capture semantic relationships between words.

### 4. LSTM & Bidirectional LSTM:
- **LSTM** units can learn long-term dependencies by maintaining a state that persists across time steps.
- **Bidirectional LSTM** processes the input in both directions—left-to-right and right-to-left—improving context understanding, especially for complex texts.

### 5. Dense Output Layer:
The final output of the model is a probability distribution over the vocabulary, where each word's probability corresponds to its likelihood of being the next word in the sequence.

---

## Model Training and Limitations

**Important Note**: Due to system constraints, the model in this project was not trained for the optimal number of epochs. Running the model for more epochs on a more powerful system will lead to better accuracy and performance. If you have access to more computational resources, you are encouraged to increase the number of training epochs for better results.

- **Current Epochs**: 20 epochs with **EarlyStopping** and **ReduceLROnPlateau** applied.
- **Suggested Epochs**: Depending on system capabilities, training for 50-100 epochs is recommended for improved accuracy.

---

## Usage Instructions

### Training the Model
To train the model with the provided data:
```bash
python train_model.py
```

### Predicting the Next Word
Use the trained model to predict the next word in a sequence:
```python
next_word = predict_next_word(model, tokenizer, "input text", max_sequence_len)
```

### Saving the Model & Tokenizer
- **Model**: Saved as `next_word_lstm.h5` using the `save` method in Keras.
- **Tokenizer**: Saved as `tokenizer.pickle` using Python's `pickle` module.

---

## Example Predictions

Here’s an example of how the model works:

**Input Text**: "Summer breeze"  
**Predicted Next Word**: "flows"

The model uses its trained understanding of word sequences to predict the next logical word in the given text.

---

## Requirements and Installation

### Requirements:
- **Python 3.x**
- **TensorFlow** and **Keras**
- **NumPy**
- **NLTK** and **Gutenberg Corpus**
- **Scikit-learn**

### Installation:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/text-gen-lstm.git
   ```
2. Navigate to the project directory:
   ```bash
   cd text-gen-lstm
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
