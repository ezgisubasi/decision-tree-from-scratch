# decision-tree-from-scratch

The Decision Tree algorithm is a problem/decision based supervised algorithm that has a tree structure and includes all possible outcomes depending on the conditions. In this program, the decision tree classifier written from scratch and scikit-learn version applied to the Titanic dataset. The decision tree algorithm written from scratch includes entropy and gini index in order to use calculate homogeneity as options, as well as max_depth and min_samples for tuning hyperparameters. Before the comparing scratch algorithm with scikit-learn, the dataset is made ready by passing through one-hot encoding for the aim of using in scikit-learn version.

<p align="center"> 
  <img width="600" alt="Ekran Resmi 2021-06-11 22 25 37" src="https://user-images.githubusercontent.com/52889449/121766540-4846e400-cb5b-11eb-829d-b2367233eb28.png">
</p>

# Divide and Conquer

The decision tree algorithm has a recursive structure. This structure is also called divide and conquer because it divides the data into subsets and continues this process over and over again until it finds the smallest subset. When the stopping criteria are met or the subset is sufficiently homogeneous, the tree stops forming. The algorithm looks at each interval to find the best split during the split process and selects the most homogeneous part. Thus, that interval turns into the node where the condition is located. 

<p align="center"> 
  <img width="500" alt="Ekran Resmi 2021-06-11 22 25 37" src="https://user-images.githubusercontent.com/52889449/121767436-dffb0100-cb60-11eb-86a1-e1bf4430bdc1.png">
</p>

# Difference between Entropy and Gini Index

Gini index and entropy are two separate methods used to measure homogeneity. The biggest difference is that gini values in the range of 0-0.5, while entropy values in the range of 0-1. Entropy is more complex because it uses logarithms in its formula, so gini index works faster.

## Entropy:

Formula;

<p align="center"> 
    <img width="350" alt="Ekran Resmi 2021-06-12 09 27 14" src="https://user-images.githubusercontent.com/52889449/121767375-7b3fa680-cb60-11eb-830b-f2b8073eba09.png">
</p>

Max and Min Values;

<p align="center"> 
    <img width="435" alt="Ekran Resmi 2021-06-12 09 27 35" src="https://user-images.githubusercontent.com/52889449/121767376-7e3a9700-cb60-11eb-97f1-a7d567535df1.png">
</p>


## Gini Index:

Formula;

<p align="center"> 
  <img width="350" alt="Ekran Resmi 2021-06-12 09 27 06" src="https://user-images.githubusercontent.com/52889449/121767386-88f52c00-cb60-11eb-8cf4-987c5ba2e11c.png">
</p>

Max and Min Values;

<p align="center"> 
  <img width="435" alt="Ekran Resmi 2021-06-12 09 27 43" src="https://user-images.githubusercontent.com/52889449/121767389-8b578600-cb60-11eb-89e5-5f7774762ff3.png">
</p>

## Information Gain:

Entropy or gini index calculates homogeneity in a given range, while information gain considers the effect of the split in each probability feature on the entire dataset by calculating the change in the homogeneity that would result from a split on each possible feature.

Formula; 

<p align="center"> 
  <img width="372" alt="Ekran Resmi 2021-06-12 09 37 44" src="https://user-images.githubusercontent.com/52889449/121767860-a8418880-cb63-11eb-8c43-642cd1a99b70.png">
</p>

Low vs High Information Gain;

<p align="center"> 
  <img width="531" alt="Ekran Resmi 2021-06-12 09 37 32" src="https://user-images.githubusercontent.com/52889449/121767861-a972b580-cb63-11eb-886e-4ae05f78eece.png">
</p>

# Hyperparameters: max_depth and min_samples

* max_depth:  The theoretical maximum depth a decision tree can achieve is one less than the number of training samples, but no algorithm will let you reach this point for obvious reasons, one big reason being overfitting. Therefore this parameter determines the maximum depth of the tree in order to prevent overfitting. 

* min_samples: This is the minimum number of samples, or data points, that are required to be present in the leaf node. The leaf node is the last node of the tree.

# Outputs of the given dataset

When we look at these graphs as a result of tuning hyperparameters, the highest accuracy is obtained for both gini index and entropy while max_depth is 45. On the other hand, when min_samples is 14 and above, the highest accuracy is also obtained for two criteria.

<p align="center"> 
  <img width="500" alt="Ekran Resmi 2021-06-11 22 25 37" src="https://user-images.githubusercontent.com/52889449/121742596-b44c2c80-cb08-11eb-9fb5-0b56aeb5f93a.png">
<img width="500" alt="Ekran Resmi 2021-06-11 22 26 10" src="https://user-images.githubusercontent.com/52889449/121742622-badaa400-cb08-11eb-9a88-fd80f194464f.png">
</p>

# Comparison with Scikit-Learn

When the decision tree algorithm written from scratch is compared with the scikit-learn version, it is observed that the accuracy of scikit-learn is higher for both criteria. This may be because the data passed through one-hot encoding before running the scikit-learn algorithm.

<p align="center"> 
  <img width="568" alt="scikit-vs-scratch" src="https://user-images.githubusercontent.com/52889449/121765903-f8661e00-cb56-11eb-8bd4-e8f90144501f.png">  
</p>
