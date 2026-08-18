# DATA110 Exam Numerical Practice

## 1. K-Means
For each point, calculate Euclidean distance to each centroid. Assign the point to the nearest centroid. Then calculate the mean of each cluster to get the new centroid. Repeat for the required iterations. If assignments/centroids stop changing, the algorithm has converged.

## 2. KNN
Calculate distance from the new point to every training point. Sort by distance. Take the nearest K observations and use majority voting for classification. Odd K reduces the chance of a tie.

## 3. Decision Tree
Entropy measures impurity. Information gain measures how much a split reduces entropy:

`Information Gain = Entropy(parent) - weighted entropy(children)`

Choose the split with the highest information gain (for the instructor's treatment).

## 4. PCA (2x2)
For a 2x2 covariance matrix, solve `det(A - lambda I) = 0` for eigenvalues. The eigenvector associated with the largest eigenvalue gives the first principal direction.

## 5. ANN Weight Update
`W_new = W_old - learning_rate * (dLoss/dW)`

Example: `0.50 - 0.10*0.20 = 0.48`.

## 6. SVM
Support vectors are the observations closest to the separating hyperplane. The margin is the separation around the hyperplane; soft-margin SVM allows some violations, controlled by C.
