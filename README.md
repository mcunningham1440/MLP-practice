# About
This script was a follow-up on my KNN practice program, using the same breast cancer dataset (for more info on that data, see the readme for knn-practice). For this program, I used a simple multilayer perceptron written in PyTorch. Rather than identifying the optimal number of principal components to use for dimensionality reduction as I did previously, here I train 3 neural networks using either

1) only the pam50 genes used for typing breast cancer (minus 2 which are not found in the dataset)
2) only the best 48 genes by ANOVA F-value
3) the top 48 principal components of the data

and compare their performance at correctly classifying the tumors in the dataset. Subsequently, I ensemble all of their predictions by averaging the percent probabilites for each class predicted by the models, and assess the accuracy of the ensemble.

# The algorithm
The neural network used for each model is a basic multilayer perceptron with one hidden layer, an architecture which I found gave the best results; adding additional layers did not appear to significantly boost performance. Although I did not do extensive and systematic hyperparameter optimization, I tweaked a few of the parameters to correct a persistent problem I experienced with training instability for the ANOVA and PCA models.

* Using the logistic sigmoid as the activation function for the hidden layer appeared to yield better performance in all 3 networks than ReLU.
* Stochastic and minibatch gradient descent both resulted in high instability of the training and test losses between epochs, which was largely rectified by using batch gradient descent.
* Adam optimization resulted in faster convergence and better performance than SGD.
* Saving a copy of the model when the average test loss over the previous 5 epochs was higher than the average over the 5 epochs before appeared to be effective at capturing the bottom of the test loss curve.

# Results
I find that, in general, the pam50 model slightly outperforms the PCA model, while both are superior to the ANOVA model. However, the ensemble predictions nearly always outperform all 3 constituent networks. This finding is consistent with the conventional wisdom in machine learning that training multiple models with different approaches to classifying a target dataset tends to yield superior results to any single model.
