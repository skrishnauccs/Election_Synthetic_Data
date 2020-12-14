# Trend Based Election Synthetic Data

A common problem in a data set with imbalanced classes is that there are too few examples of the minority class for a model to learn the decision boundary effectively. This problem can be solved by increasing the samples in the minority class. The most widely used approach for synthesizing new examples is SMOTE, proposed by Nitesh Chawla et al.<br/>
SMOTE goes beyond simple under or oversampling. This algorithm creates new instances of the minority class by creating convex combinations of neighboring instances. Python’s Imbalanced-Learn Library is used to implement SMOTE <br/>
SMOTE works by selecting examples close in the feature space, drawing a line between the examples in the feature space, and drawing a new sample at a point along that line. The below example demonstrates how SMOTE works.<br/>


# Results & Analysis
# Benchmark Results
<img src="Images/BenchmarkResults/DT.png" width='300px' height ='300px' > <img src="Images/BenchmarkResults/RF.png" width='300px' height ='300px' > 
<img src="Images/BenchmarkResults/KNN.png" width='300px' height ='300px' > <img src="Images/BenchmarkResults/SVM.png" width='300px' height ='300px' >
<img src="Images/BenchmarkResults/NB.png" width='300px' height ='300px' > <img src="Images/BenchmarkResults/XGBoost.png" width='300px' height ='300px' >

# SMOTE Results
<img src="Images/Smoteresults/DT.png" width='300px' height ='300px' > <img src="Images/Smoteresults/RF.png" width='300px' height ='300px' > 
<img src="Images/Smoteresults/KNN.png" width='300px' height ='300px' > <img src="Images/Smoteresults/SVM.png" width='300px' height ='300px' >
<img src="Images/Smoteresults/NB.png" width='300px' height ='300px' > <img src="Images/Smoteresults/XGboost.png" width='300px' height ='300px' >


# SMOTE with hyperparameter tuning Results
<img src="Images/HyperparamterTuning/DT.png" width='300px' height ='300px' > <img src="Images/HyperparamterTuning/RF.png" width='300px' height ='300px' > 
<img src="Images/HyperparamterTuning/KNN.png" width='300px' height ='300px' > <img src="Images/HyperparamterTuning/SVM.png" width='300px' height ='300px' >
<img src="Images/HyperparamterTuning/NB.png" width='300px' height ='300px' > <img src="Images/HyperparamterTuning/XGBoost.png" width='300px' height ='300px' >

# Comparitive Anlaysis
<img src="Images/Comparitiveresults/Final_accuracy.png" width='400px' height ='400px' > <img src="Images/Comparitiveresults/Final_recall.png" width='400px' height ='400px' > 

<img src="Images/Comparitiveresults/Final_precision.png" width='400px' height ='400px' > <img src="Images/Comparitiveresults/Final_f1.png" width='400px' height ='400px' >

