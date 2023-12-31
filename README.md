# Stock Market Forecasting using DRAGAN and Feature Matching

This repository contains the code implementation for the paper titled "Stock market forecasting using DRAGAN and feature matching". 

## Introduction

Applying machine learning methods to forecast stock prices has been a topic of interest in recent years. However, a few studies have been reported based on generative adversarial networks (GANs) in this area, but their results are promising. While GANs are powerful generative models successfully applied in different areas, they suffer from inherent challenges such as training instability and mode collapse. Another primary concern is capturing correlations in stock prices. Therefore, the main challenges fall into two categories: capturing correlations and addressing the inherent problems of GANs. In this paper, we introduce a novel framework based on DRAGAN  and feature matching for stock price forecasting, which improves training stability and alleviates mode collapse. We employ windowing to acquire temporal correlations by the generator and exploit conditioning on discriminator inputs to capture temporal correlations and correlations between prices and features. Experimental results on data from several stocks indicate that proposed method outperforms long short-term memory (LSTM) as a baseline method, as well as basic GANs and WGAN-GP  as two different variants of GANs. 

This repository contains the Python source code for the paper:tock market forecasting using DRAGAN and feature matching


## Repository Structure

- `Data/`: This directory contains the stocks datasets used for training and testing.
- `Model_trained/`: This directory contains trained models.

## Requirements

python 

tensorflow

ta library [https://github.com/bukosabino/ta]

We appreciate the efforts of the TA developers, which is a Technical Analysis library for feature engineering from financial time series datasets (Open, Close, High, Low, Volume). It is built on Pandas and Numpy.

## Contributing

We welcome contributions to enhance the functionality and performance of the models. If you have any suggestions, bug reports, or feature requests, please contact me.

## Citation
Nejad, F. S., & Ebadzadeh, M. M. (2024). Stock market forecasting using DRAGAN and feature matching. Expert Systems with Applications, 244, 122952. https://doi.org/10.1016/j.eswa.2023.122952


## Contact
        
For any inquiries or further information, please contact [fa.shahabi@aut.ac.ir]
