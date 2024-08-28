# Estimating Galaxy Distances with Advanced Regression Models

This project focuses on developing predictive models to estimate the z_spec variable, a spectroscopic redshift measurement, using various features from the Happy and Teddy datasets. The goal is to apply different machine learning models to identify which model performs best in predicting z_spec based on selected features. The models used include Random Forest, Gradient Boosting, Neural Network, Decision Tree, and XGBoost.
The project is structured around training the models on a subset of the datasets (Happy A and Teddy A) and evaluating their performance on the remaining subsets (Happy B, C, D, and Teddy B, C, D). This approach allows for a thorough evaluation of the model's predictive capabilities across different datasets that contain slightly varying features.

**Dataset Link**
https://github.com/COINtoolbox/photoz_catalogues.

**Data Structure**
The data used in this project is derived from two main sources: the Happy dataset and the Teddy dataset. Each dataset is divided into four parts: A, B, C, and D.

**Happy Dataset:**
•	Happy A, B, C, D: These files contain various features including mag_r, u-g, g-r, r-i, i-z, and the target variable z_spec.
•	Features:
	mag_r: Magnitude in the r-band.
	u-g: Color index calculated by subtracting the g-band magnitude from the u-band magnitude.
	g-r, r-i, i-z: Similar color indices for other bands.
	z_spec: The target variable, spectroscopic redshift.
	feat1 to feat5: Normalized versions of the above features, used for model training.

**Teddy Dataset:**
•	Teddy A, B, C, D: These files follow a similar structure to the Happy datasets, containing the same set of features.
Each part (A, B, C, D) is used to evaluate the model's performance under slightly different conditions, as the features can vary across these subsets.


**Preprocessing Steps**

Before model training, the datasets undergo several preprocessing steps to ensure the models can learn effectively from the data:

1.	Data Cleaning: The datasets are first checked for missing values. Any rows containing missing data are dropped to ensure the integrity of the training and testing process. This step is crucial to avoid introducing bias or errors in the model training phase.

2.	Feature Selection: The features selected for model training include feat1 to feat5, which are normalized versions of the original features such as mag_r, u-g, etc. These normalized features are chosen because they provide a more stable input for the machine learning models, leading to potentially better performance.

3.	Normalization: The selected features (feat1 to feat5) are standardized using a StandardScaler. This process transforms the features so that they have a mean of 0 and a standard deviation of 1. Normalization is an essential step, especially for models like neural networks and gradient boosting, which are sensitive to the scale of the input data.

4.	Saving Preprocessed Data: After preprocessing, the cleaned and normalized datasets are saved for further analysis. These saved datasets are then used as input for the model training and evaluation process.
This structured approach ensures that the data fed into the models is clean, relevant, and standardized, thereby enhancing the reliability and accuracy of the predictive models.

**Model Details**
The project involves the application of several machine learning models to predict the spectroscopic redshift (z_spec) based on a set of features from the Happy and Teddy datasets. The models employed in this project include:

**1.	Random Forest Regressor:**
•	Description: Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mean prediction of the individual trees. It is robust against overfitting and is effective in handling large datasets with higher dimensionality.
•	Hyperparameters:
	n_estimators: Number of trees in the forest (tuned with values 50, 100, 200).
	max_features: Number of features to consider at each split (tuned with 'sqrt' and 'log2').
	max_depth: Maximum depth of the tree (tuned with values None, 10, 20, 30).

**2.	Gradient Boosting Regressor:**
•	Description: Gradient Boosting is an ensemble technique that builds models sequentially, each new model attempting to correct the errors made by the previous ones. It is known for its high accuracy but can be prone to overfitting if not properly tuned.
•	Hyperparameters:
	n_estimators: Number of boosting stages (tuned with values 50, 100, 200).
	learning_rate: Shrinks the contribution of each tree (tuned with values 0.01, 0.1, 0.2).
	max_depth: Maximum depth of the individual estimators (tuned with values 3, 4, 5).

**3.	Neural Network (MLPRegressor):**
•	Description: The MLPRegressor is a feedforward artificial neural network model. It can capture complex non-linear relationships in the data. The network used in this project consists of a single hidden layer with 100 neurons, utilizing the ReLU activation function and the Adam optimizer.
•	Hyperparameters:
	hidden_layer_sizes: Size of the hidden layers (set to (100,)).
	activation: Activation function for the hidden layer (set to 'relu').
	solver: Optimizer (set to 'adam').
	max_iter: Maximum number of iterations (set to 500).

**4.	Decision Tree Regressor:**
•	Description: Decision Tree Regressor is a simple, interpretable model that splits the data into subsets based on feature values. It works well for capturing non-linear patterns but can overfit if not pruned.
•	Hyperparameters:
	No specific tuning applied; the default settings are used to maintain simplicity and interpretability.

**5.	XGBoost Regressor:**
•	Description: XGBoost is an optimized implementation of gradient boosting. It is designed for efficiency, speed, and performance. XGBoost is particularly useful for large datasets and complex feature spaces.
•	Hyperparameters:
	n_estimators: Number of boosting rounds (tuned with values 50, 100, 200).
	learning_rate: Controls the contribution of each tree (tuned with values 0.01, 0.1, 0.2).
	max_depth: Maximum depth of a tree (tuned with values 3, 4, 5).
Each model is trained using the Happy A and Teddy A datasets and then evaluated on the Happy B, Happy C, Happy D, Teddy B, Teddy C, and Teddy D datasets. The evaluation metrics include Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and R-squared (R^2) values. These metrics are used to compare the performance of the models across the different test sets, helping identify the model that best generalizes to unseen data.


#**Usage Instructions**
To use the code and models provided in this project, follow these steps:
1.	Clone the Repository:
•	Use the following command to clone the repository
git clone https://github.com/username/repository-name.git
•	Navigate into the project directory
cd repository-name
2.	Install Dependencies:
•	Ensure you have Python 3.7 or above installed. Install the required packages using the following command:
pip install -r requirements.txt
•	The requirements.txt file includes all necessary packages such as scikit-learn, xgboost, matplotlib, pandas, and numpy.
3.	Running the Preprocessing Script:
•	To preprocess the datasets (cleaning, feature selection, and normalization), run:
Jupyter Notebook Source File
•	The script in that file will generate cleaned and scaled datasets that will be saved as CSV files for subsequent model training.
4.	Training the Models:
•	To train the models, run:
Jupyter Notebook Source File
•	The script in that file will load the pre-processed data, perform hyperparameter tuning where applicable, train the models, and save the best-performing models as .pkl files.
5.	Evaluating the Models:
•	To evaluate the models on the test datasets (Happy B, Happy C, Happy D, Teddy B, Teddy C, Teddy D), run:
Jupyter Notebook Source File
•	The script in that file will generate evaluation metrics and visualizations such as scatter plots for true vs. predicted values and save these results for further analysis.


**Conclusion**
This project demonstrates the application of various machine learning models to predict spectroscopic redshift (z_spec) using features from the Happy and Teddy datasets. The project involved thorough data preprocessing, model training, and evaluation across multiple datasets to ensure the robustness and generalizability of the models. The results indicate the strengths and limitations of each model, with metrics such as RMSE, MAE, and R-squared providing a quantitative assessment of performance.
By following the usage instructions, users can replicate the results, apply the models to new data, and extend the analysis further. The detailed preprocessing steps and model descriptions ensure that others can understand, utilize, and build upon this work effectively. The inclusion of EDA plots aids in comprehending the underlying structure of the data, which is crucial for informed model development and interpretation of results.
This project serves as a comprehensive example of how to approach predictive modeling in a structured and methodical manner, making it a valuable resource for both practitioners and researchers in the field of data science and machine learning.


