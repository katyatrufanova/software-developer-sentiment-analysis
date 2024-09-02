# Software Developer Sentiment Analysis

This project implements sentiment analysis classifiers trained on software developers' communication channels, focusing on state-of-the-art research in sentiment analysis, particularly in the context of software engineering.

## Dataset

The dataset used for this project is publicly available in the [Senti4SD repository](https://github.com/collab-uniba/Senti4SD), specifically in the [Senti4SD_Train_Test_Partitions](https://github.com/collab-uniba/Senti4SD/tree/master/Senti4SD_GoldStandard_and_DSM/Senti4SD_Train_Test_Partitions) folder. This dataset contains over 4K posts extracted from Stack Overflow, annotated for sentiment polarity.

## Project Structure

The project consists of two Jupyter notebooks that can be run in Google Colab:

1. `DistilBERT.ipynb`: Implements a DistilBERT-based model for sentiment analysis.
2. `biLSTM.ipynb`: Implements bidirectional LSTM models, both with and without GloVe embeddings.

## Models and Experiments

### DistilBERT
DistilBERT is a smaller version of BERT that retains most of its performance while being faster and requiring less memory. The notebook includes:

- Text Cleaning: Removal of HTML tags, digits, punctuation, stopwords, and conversion to lowercase.
- Tokenization: Using DistilBERT tokenizer with padding and truncation.
- Model Architecture: Pre-trained DistilBERT model with a dense layer and softmax activation.
- Training: 80% training, 20% validation, maximum 5 epochs, batch size 16, early stopping, Adam optimizer with 5e-5 learning rate.

### Bidirectional LSTM
Two experiments were conducted with bidirectional LSTM:

1. With GloVe embeddings
2. Without GloVe embeddings

Both experiments include:

- Text Cleaning and Tokenization using Keras' Tokenizer
- Bidirectional LSTM layer with 128 units
- Dense layer with softmax activation
- Training: Adam optimizer, 5e-5 learning rate, batch size 16, 5 epochs, early stopping

The GloVe experiment uses pre-trained word embeddings from Twitter, Wikipedia, and Common Crawl corpus.

## Results

F1-scores:
1. DistilBERT: 0.85
2. BiLSTM with GloVe: 0.83
3. BiLSTM without GloVe: 0.78

The DistilBERT model achieved the best performance.

## Error Analysis

Error analysis was conducted on a portion of the validation set for the DistilBERT model. Key observations include:

1. Misclassification of neutral sentiments
2. Confusion between neutral and negative sentiments
3. Difficulty with complex sentences
4. Lack of domain-specific knowledge

## Future Work

Potential improvements:
- Increase the diversity of the training data
- Use a larger dataset
- Further hyperparameter tuning

## Usage

To use this project:

1. Clone this repository to your local machine or directly to your Google Drive.
2. Open the notebooks in Google Colab.
3. Run the cells in each notebook sequentially to perform the sentiment analysis.

Note: Make sure to adjust the file paths in the notebooks to match the location of the dataset in your Google Drive or local environment.

## Requirements

The project is designed to run in Google Colab, which provides all necessary libraries. If you wish to run the notebooks locally, you'll need to install the required Python libraries (such as transformers, tensorflow, and keras).

## Acknowledgements

This project uses the dataset from the Senti4SD project. If you use this dataset in your research, please cite the following paper:

Calefato, F., Lanubile, F., Maiorano, F., Novielli N. (2018) "Sentiment Polarity Detection for Software Development," Empirical Software Engineering, 23(3), pp:1352-1382, doi: https://doi.org/10.1007/s10664-017-9546-9.