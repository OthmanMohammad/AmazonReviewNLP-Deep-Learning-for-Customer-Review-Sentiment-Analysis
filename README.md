# AmazonReviewNLP: Sentiment Analysis for Customer Reviews

AmazonReviewNLP is a deep learning project that utilizes LSTMs for sentiment analysis on Amazon customer reviews. The project is structured with two main components: `utils.py` and `train.py`.

## Dataset

The dataset used in this project can be found on Kaggle: [Amazon Reviews](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews). It consists of a few million Amazon customer reviews (input text) and star ratings (output labels) for learning how to train fastText for sentiment analysis.

## Model Architecture

The model uses a bidirectional LSTM architecture for sentiment analysis. The main components of the model are:

1. An Embedding layer for converting input text into dense vectors.
2. Two Bidirectional LSTM layers with dropout and recurrent dropout to capture both forward and backward context.
3. A final LSTM layer for sequence processing.
4. A Dense layer with a softmax activation function for output classification.

## Usage

To train the model, run the following command:

```shell 
python train.py 
```


## Downsampling

If you find that the dataset is too large to train, you can downsample the data by modifying the `data_generator` function. 

1. Add a new parameter `downsample_frac` to the `data_generator` function in `utils.py`. This parameter will control the fraction of data to be used during training and validation.

2. Update the `data_generator` function to downsample the data using the `random.sample()` function with the `downsample_frac` parameter.
```python
import random

def data_generator(file_path, batch_size, max_features, max_len, downsample_frac=1.0):
    while True:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Apply downsampling
            if downsample_frac < 1.0:
                lines = random.sample(lines, int(len(lines) * downsample_frac))
            ...
```

3. Pass the `downsample_frac` parameter when calling the `data_generator` function in `train.py`. Adjust the value of `downsample_frac` as needed.

For example, to use 10% of the data, set `downsample_frac = 0.1`.

```python
downsample_frac = 0.1  # Adjust this value as needed
train_generator = data_generator("data/train.ft.txt", train_batch_size, max_features, max_len, downsample_frac)
val_generator = data_generator("data/test.ft.txt", val_batch_size, max_features, max_len, downsample_frac)
```

## Contributing

Contributions are welcome. Please submit a pull request or open an issue to discuss proposed changes.

## Author

This project was developed by [Mohammad Othman](https://github.com/OthmanMohammad). Feel free to reach out for any questions or suggestions related to the project.

