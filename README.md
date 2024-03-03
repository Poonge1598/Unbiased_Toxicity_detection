# Unbiased Toxicity Detection with Custom Loss Function

## Project Overview

This project introduces an advanced approach to detecting toxic comments on online platforms, emphasizing unbiased detection through the use of a custom loss function and innovative deep learning models.

## Introduction

Detecting toxic behavior online is essential for maintaining positive user interactions. Conventional methods often suffer from biases, leading to misidentification and unfair targeting. Our solution employs two sophisticated models to accurately and fairly detect toxic content, utilizing state-of-the-art NLP techniques and deep learning.

## Methodology

### Data Preprocessing

We preprocess our dataset through a series of NLP techniques, including tokenization, stop-word removal, and normalization, alongside specific strategies aimed at reducing bias in the training data.

### Creating the Vocab for Embedding Matrix

Utilizing GloVe embeddings, we create an embedding matrix tailored to our dataset, enabling our models to understand language semantics by mapping words to vectors in a multidimensional space.

### Model 1: Attention-Based Neural Network

Model 1 is an attention-based neural network that leverages LSTM and GRU layers, enhanced with a custom attention mechanism, to effectively process text data for toxicity detection. The model architecture includes:

- **Embedding Layer**: Incorporates a pre-trained embedding matrix for vectorization.
- **LSTM Layer**: A bidirectional LSTM layer for capturing context from both text directions.
- **GRU Layer**: A bidirectional GRU layer to further refine the text analysis.
- **Attention Layer**: A custom attention mechanism to focus on relevant text segments.
- **Fully Connected Layers**: Linear and ReLU activations interpret the attention output, leading to a prediction layer.

### Model 2: GloVeClassifier with Multi-head Attention

Model 2, `GloVeClassifier_mul_att`, introduces a multi-head attention mechanism to further enhance the model's ability to focus on various parts of the text simultaneously, improving accuracy in detecting toxicity. Key features include:

- **Multi-head Attention**: Divides the attention mechanism into multiple heads, allowing the model to attend to different parts of the sequence for a comprehensive understanding.
- **Bidirectional LSTM and GRU Layers**: Capture sequential information and dependencies, addressing the vanishing gradient problem.
- **Fully Connected Layers**: Employ linear layers and LeakyReLU activation to process the attention mechanism's output for final classification.

### Custom Loss Function

Both models utilize a custom loss function designed to minimize bias in toxicity detection. This function penalizes predictions that exhibit bias, encouraging the model to identify patterns genuinely indicative of toxicity.

## Results

The implementation of these models demonstrates a significant improvement over traditional methods in both accuracy and fairness. Model 2, with its multi-head attention mechanism, shows a notable increase in performance, effectively identifying toxic comments without relying on biased indicators.

## Conclusion

This project represents a significant advancement in creating fair and effective online toxicity detection systems. By integrating sophisticated model architectures with a custom loss function, we have developed a robust framework that significantly reduces bias in toxicity detection. Future work will explore further enhancements and applications of this technology.

