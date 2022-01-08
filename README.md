# Binary-Classifier-Training-to-predict-annual-training
The project consists of a binary classification task
to predict the annual income of individuals. There are
several machine learning algorithms which complete
this. In our approach we tested different algorithms to
calculate a baseline score for the method and afterwards
we chose the algorithm with the highest accuracy. We
used stratified k fold cross-validation [11] procedure to
evaluate model performance. Due to the high skewness
of the data a stratified approach better represented the
data selection to train the model.
The algorithms tested are:

Support Vector Machine 
Random Forest Classifier 
Bagging Classifier
Gradient Boosting Classifier

The algorithm with the highest score was gradient
boosting with the cross validation score of 0.865 and stan-
dard deviation of (0.005)
