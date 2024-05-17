# A Federated Data Fusion-Based Prognostic Model for Applications with Multi-Stream Incomplete Signals

## Description

This repository includes the code for replicating results from "A Federated Data Fusion-Based Prognostic Model
for Applications with Multi-Stream Incomplete Signals" by Madi Arabi and Xiaolei Fang. 
Codes can be used for general purpose of fedearted time to failure (TTF) prediction with multiple users. Federated learning

Industrial prognostic aims to predict the failure time of machines by analyzing their degradation signals. This is crucial for maintaining the reliability and efficiency of industrial operations. Our project proposes a federated learning approach that allows multiple organizations to collaboratively train a prognostic model without sharing their data, thus preserving data privacy. This federated prognostic model uses multi-stream degradation signals from multiple sensors to improve prediction accuracy. By employing multivariate functional principal component analysis (MFPCA) for data fusion and (log)-location-scale (LLS) regression for TTF prediction, our model effectively handles high-dimensional and incomplete data, which is common in real-world applications.

The proposed method enables organizations to use their isolated data to jointly develop a reliable prognostic model. This approach not only enhances the model's performance by leveraging diverse datasets but also ensures compliance with privacy regulations.

### Why It Is Useful

- Collaborative Model Training: Allows multiple organizations to jointly train a prognostic model using their data without compromising privacy.
- Improved Prediction Accuracy: Utilizes multi-stream degradation signals for better performance compared to single-stream models.
- Handles Incomplete Data: Capable of processing high-dimensional and incomplete degradation signals, addressing common real-world data challenges.
- Privacy-Preserving: Ensures data privacy by keeping all participants' data local and confidential.
### Project Outline
- Data Fusion: Using MFPCA to reduce the dimensionality of multi-stream degradation signals and extract essential features.
- Prognostic Model Construction: Employing LLS regression to map TTFs to the extracted features.
- Federated Learning Framework: Developing a federated algorithm to enable collaborative model training across multiple users while preserving data privacy.
## Requirements

| Package   | Version |
|-----------|---------|
| numpy     | 1.21.0+ |
| numba     | 0.53.0+ |
| pandas    | 1.3.0+  |
| sklearn   | 0.24.0+ |
| math      | Built-in |
| time      | Built-in |

To install the required packages, run:

pip install numpy numba pandas scikit-learn

## Usage

The code snippets are performed on simulated data using "Multi-sensor prognostics modeling for applications with highly incomplete signals.". For any other application, the input data can be replaced. 

## License

This project is licensed under the MIT License

## References

- Fang, X., H. Yan, N. Gebraeel, and K. Paynabar (2021). Multi-sensor prognostics modeling for applications with highly incomplete signals. IISE Transactions 53 (5), 597–613.
