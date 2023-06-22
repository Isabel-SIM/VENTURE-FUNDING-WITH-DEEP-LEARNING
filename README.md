<h1>Venture Funding with Deep Learning</h1>

In this project, I have successfully developed a deep learning model to predict the success of startup funding applications for Alphabet Soup, a venture capital firm. The goal was to create a binary classifier model that could determine whether an applicant would become a successful business if funded by Alphabet Soup.<br><br>

In order to achieve this, I followed the approach outlined in the instructions:<br>

<h5>Data Preparation:</h5>

- I started by analysing a CSV file containing more than 34,000 organisations that received funding from Alphabet Soup. I identified categorical variables that needed encoding and columns that defined the features and target variables.<br>
- I dropped irrelevant columns like "EIN" (Employer Identification Number) and "NAME" from the DataFrame.<br>
- Using OneHotEncoder, I encoded the categorical variables and created a new DataFrame to store the encoded variables.<br>
- I combined the numerical variables from the original DataFrame with the encoded variables using the concat() function in Pandas.<br>


<h5>Neural Network Model:</h5>

- Using TensorFlow and Keras, I designed a deep neural network model for binary classification. I determined the number of input features, layers, and neurons on each layer based on the dataset.<br>
- I compiled and fit the model using the <i>binary_crossentropy loss</i> function, the <i>adam optimiser,</i> and the <i>accuracy evaluation metric.</i> <br>
- The model was evaluated using test data to calculate its loss and accuracy.<br>
- Finally, I saved and exported the trained model to an HDF5 file.<br>


<h5>Model Optimisation:</h5>

- I made two additional attempts to optimise the model and improve its accuracy. For each attempt, I defined a new deep neural network model and experimented with different techniques.<br>
- These techniques included adjusting input data, adding more/less neurons and hidden layers, using different activation functions and altering the number of training epochs.<br>
- I compared the accuracy scores achieved by each model to assess their performance.<br>
- Throughout the project, I utilised various libraries such as Pandas, scikit-learn, TensorFlow, and Keras to preprocess the data, build the neural network model, and evaluate its performance. By applying machine learning and deep learning techniques, I have created a predictive model that can assist Alphabet Soup in making informed decisions regarding startup funding applications.
<br><br><br>
I aimed to achieve a better accuracy with each model, which I managed to do.
