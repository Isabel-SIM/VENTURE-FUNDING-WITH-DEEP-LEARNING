# Venture Funding with Deep Learning

## Project Overview

In this project, I have successfully developed a deep learning model to predict the success of startup funding applications for Alphabet Soup, a venture capital firm. The goal was to create a binary classifier model that could determine whether an applicant would become a successful business if funded by Alphabet Soup.

## Approach and Steps

### Data Preparation

I started by analyzing a CSV file containing more than 34,000 organizations that received funding from Alphabet Soup. The following data preparation steps were undertaken:

1. **Data Exploration**: I reviewed the DataFrame, identifying categorical variables that needed encoding and columns that would define the features and target variables.

2. **Feature Selection**: I dropped irrelevant columns like "EIN" (Employer Identification Number) and "NAME" from the DataFrame as they were not relevant to the binary classification model.

3. **Encoding Categorical Data**: Using the `OneHotEncoder`, I encoded the categorical variables and created a new DataFrame to store the encoded variables.

4. **Combining Features**: I combined the numerical variables from the original DataFrame with the encoded variables using the `concat()` function in Pandas.

### Neural Network Model

Using TensorFlow and Keras, I designed a deep neural network model for binary classification. The key steps were:

1. **Model Architecture**: I determined the number of input features, layers, and neurons on each layer based on the dataset. I initially started with a two-layer deep neural network model using the `relu` activation function for both layers.

2. **Model Compilation**: I compiled the model using the `binary_crossentropy` loss function, the `adam` optimizer, and the `accuracy` evaluation metric.

3. **Model Training**: The model was trained using the training data to optimize its parameters.

4. **Model Evaluation**: The model was evaluated using test data to calculate its loss and accuracy.

5. **Model Export**: Finally, I saved and exported the trained model to an HDF5 file named `AlphabetSoup.h5`.

### Model Optimization

I made two additional attempts to optimize the model and improve its accuracy. For each attempt, I experimented with various techniques, including:

- Adjusting input data by dropping different feature columns.
- Adding more neurons and hidden layers.
- Using different activation functions for hidden layers.
- Altering the number of training epochs.

The objective was to enhance the model's predictive accuracy.

## Project Results

Throughout the project, I utilized various libraries such as Pandas, scikit-learn, TensorFlow, and Keras to preprocess the data, build the neural network model, and evaluate its performance. By applying machine learning and deep learning techniques, I have created a predictive model that can assist Alphabet Soup in making informed decisions regarding startup funding applications.

### Model Performance

- Original Model Results: Accuracy of approximately 72.76%
- Alternative Model 1 Results: Accuracy of approximately 73.19%
- Alternative Model 2 Results: Accuracy of approximately 73.31%

The goal was to achieve better accuracy with each model, and I successfully improved the model's performance through optimization attempts.

## Instructions for Running the Code

The project consists of three main sections:

1. Data Preparation
2. Building and Training the Neural Network Model
3. Model Optimization

Each section includes detailed code and explanations. You can follow the instructions provided in the project notebook to reproduce the results.
