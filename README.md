# Stock Market Forecasting using DRAGAN and Feature Matching

This repository contains the code implementation for the paper titled "Stock market forecasting using DRAGAN and feature matching". 

## Introduction

Applying machine learning methods to forecast stock prices has been a topic of interest in recent years. However, a few studies have been reported based on generative adversarial networks (GANs) in this area, but their results are promising. While GANs are powerful generative models successfully applied in different areas, they suffer from inherent challenges such as training instability and mode collapse. Another primary concern is capturing correlations in stock prices. Therefore, the main challenges fall into two categories: capturing correlations and addressing the inherent problems of GANs. In this paper, we introduce a novel framework based on DRAGAN  and feature matching for stock price forecasting, which improves training stability and alleviates mode collapse. We employ windowing to acquire temporal correlations by the generator and exploit conditioning on discriminator inputs to capture temporal correlations and correlations between prices and features. Experimental results on data from several stocks indicate that proposed method outperforms long short-term memory (LSTM) as a baseline method, as well as basic GANs and WGAN-GP  as two different variants of GANs. 

## Repository Structure

- `data/`: This directory contains the stocks datasets used for training and testing.
- `models/`: This directory contains trained models.

## Requirements
python 
tensorflow
ta library 
We appreciate the efforts of the TA developers, which is a Technical Analysis library sound for feature engineering from financial time series datasets (Open, Close, High, Low, Volume). It is built on Pandas and Numpy.

## Getting Started

To get started with the code, follow these steps:

## Results

We have achieved promising results with our proposed approach, demonstrating improved accuracy in stock market forecasting compared to existing methods. For detailed analysis and findings, please refer to the paper.

## Contributing

We welcome contributions to enhance the functionality and performance of the models. If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any inquiries or further information, please contact [your-email@example.com](mailto:your-email@example.com).
