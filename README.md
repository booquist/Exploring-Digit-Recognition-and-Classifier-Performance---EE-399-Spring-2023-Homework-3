# Exploring-Digit-Recognition-and-Classifier-Performance---EE-399-Spring-2023-Homework-3
**Author:** Brendan Oquist <br>
**Abstract:** This report investigates the recognition and classification of handwritten digits using various machine learning techniques such as correlation matrices, singular value decomposition (SVD), linear discriminant analysis (LDA), support vector machines (SVM), and decision trees. We examine the well-known MNIST dataset and preprocess the images into a suitable format for analysis. By computing correlation matrices, we explore similarities and differences between digits, and uncover the underlying structure of the data through eigenvectors and SVD. Our investigation of the principal component directions emphasizes the importance of these techniques in dimensionality reduction and feature extraction for applications in computer vision and machine learning. We compare the performance of LDA, SVM, and decision tree classifiers in separating and identifying individual digits, providing insights into the strengths and weaknesses of these methods for digit recognition tasks.

## I. Introduction and Overview
This project investigates the recognition and classification of handwritten digits using various machine learning techniques such as correlation matrices, eigenvectors, singular value decomposition (SVD), linear discriminant analysis (LDA), support vector machines (SVM), and decision trees to uncover underlying structures and relationships within the data. We start by introducing the concept of correlation matrices, which provide a quantitative measure of the linear relationship between pairs of images in our dataset. We then explore eigenvectors and eigenvalues as a means of identifying key patterns in the data, highlighting their significance in understanding the underlying structure of handwritten digits. Additionally, we delve into the performance of LDA, SVM, and decision tree classifiers in separating and identifying individual digits, offering insights into the strengths and weaknesses of these methods for digit recognition tasks.

## II. Theoretical Background
In this section, we provide the necessary mathematical background for handwritten digit analysis, including correlation matrices, eigenvectors, eigenvalues, singular value decomposition (SVD), linear discriminant analysis (LDA), support vector machines (SVM), and decision trees. We also introduce the procedures we used, such as matrix operations and visualizations.

### 1. **Correlation Matrices**
A correlation matrix is a square matrix that contains the Pearson correlation coefficients between pairs of variables (or images in our case). The Pearson correlation coefficient measures the strength of the linear relationship between two variables. It is computed as:

$$r_{xy} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$,

where $x_i$ and $y_i$ are the individual data points, and $\bar{x}$ and $\bar{y}$ are the means of the respective variables.

In our context, we compute the dot product between pairs of images as the correlation, which simplifies the correlation calculation to:

$$c_{jk} = x_j^T x_k$$,

where $x_j$ and $x_k$ are columns of the matrix X.

### 2. **Eigenvectors and Eigenvalues**
Eigenvectors and eigenvalues are fundamental concepts in linear algebra that provide insights into the underlying structure of a matrix. For a given square matrix A, an eigenvector v and its corresponding eigenvalue λ satisfy the following equation:

$$A\textbf{v} = \lambda\textbf{v}$$

Eigenvectors represent the directions in which the matrix A stretches or compresses the data, while eigenvalues indicate the magnitude of that stretching or compression. In our analysis, we compute the eigenvectors and eigenvalues of the matrix Y = XX^T to identify key patterns in the handwritten digit dataset.

### 3. **Singular Value Decomposition (SVD)**
Singular value decomposition (SVD) is a factorization technique that decomposes a given matrix X into three matrices U, S, and V^T:

$$X = USV^T$$

U and V are orthogonal matrices containing the left and right singular vectors of X, respectively, while S is a diagonal matrix containing the singular values of X. SVD can be used for dimensionality reduction, feature extraction, and data compression by selecting a subset of singular vectors that capture the most significant variations in the data.

If you're still unsure of exactly what's going on in the SVD, let's think about it like this: 

Imagine you have a bunch of photos of handwritten numbers, and you want to find a way to understand the important features that make each number unique. Singular Value Decomposition, or SVD for short, is a cool math trick that helps us do just that.

Think of the photos as a big stack of papers. SVD takes that big stack and rearranges it into three smaller stacks. The first stack (let's call it U) tells us about the main patterns in the photos. The second stack (let's call it S) has information about how important each of those patterns is. And the third stack (let's call it V) tells us how each photo relates to the main patterns we found.

Now, instead of looking at every single photo, we can just look at a few of the most important patterns to understand what makes each number unique. This makes it much easier and faster to work with the photos and helps us in tasks like recognizing handwritten numbers or compressing the images to save storage space.

So, SVD is like a magic trick of linear algebra that helps us find the most important patterns in a big stack of photos, making it easier to understand and work with them.

In our analysis, we perform SVD on the handwritten digit matrix X and examine the principal component directions to uncover the underlying structure of the data.

### 4. **Matrix Operations and Visualization**
To effectively analyze the handwritten digits, we employ various matrix operations such as dot products, transposes, and decompositions. Additionally, we use visualization techniques like pcolor and imshow to display correlation matrices and SVD modes, which provide a visual understanding of the relationships between images and the significant patterns captured by the SVD modes.

## III. Algorithm Implementation and Development
In this section, we provide an overview of the code and steps taken to identify MNIST digits using various techniques.
**Loading and Preprocessing the Dataset** <br>

We begin by loading the MNIST dataset using the fetch_openml method from the sklearn.datasets library. 
The data and target variables are stored in X and y, respectively.

```
# Load MNIST dataset
mnist = fetch_openml('mnist_784')
X, y = mnist['data'].to_numpy(), mnist['target']
```

The images are then reshaped into column vectors and standardized to have a mean of zero and a standard deviation of one.
```
# Reshape images into column vectors
X_flattened = X.reshape(X.shape[0], -1)

# Standardize the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_flattened)
```

**Performing Singular Value Decomposition (SVD)** <br>
Next, we perform SVD on the standardized data using NumPy's linalg.svd method. 
We keep the first 50 dimensions (or modes) for analysis and plot the 9 most important SVD modes as "eigen-digits".

```
# Perform SVD
U, s, Vt = np.linalg.svd(X_standardized, full_matrices=False)

# Choose the number of dimensions (k) to keep
k = 50

# Plot the 9 most important SVD modes (eigen-digits)
num_modes = 9
fig, axes = plt.subplots(3, 3, figsize=(8, 8))

for i, ax in enumerate(axes.flat):
    eigen_digit = Vt[i].reshape(28, 28)
    ax.imshow(eigen_digit)
    ax.set_title(f'SVD Mode {i + 1}')
    ax.axis('off')

plt.tight_layout()
plt.show()
```
Each of the images represents an "eigen-digit", U, corresponding to one of the 9 most important modes in the MNIST dataset. <br>

We then calculate the number of modes required for good image reconstruction, which is determined by the energy retained by the modes.
In this case, we choose to retain at least 90% of the energy.
```
# Calculate the number of modes required for good image reconstruction
energy = np.sum(s**2)
cumulative_energy = np.cumsum(s**2)
r = np.argmax(cumulative_energy / energy > 0.9)
```

Next, we project the standardized data onto the selected V-modes, which in this case are modes 1, 2, and 4.
```
# Project data onto selected V-modes
selected_modes = [1, 2, 4]  # Using 1-based indexing
X_projected = X_standardized @ Vt[([mode - 1 for mode in selected_modes]), :].T
```

We also create a 3D scatter plot using the projected data to visualize the distribution of the digits in the reduced feature space.
```
# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for digit in range(10):
    indices = (y == str(digit))
    ax.scatter(X_projected[indices, 0], X_projected[indices, 1], X_projected[indices, 2], label=f'Digit {digit}')

ax.set_xlabel(f'V-mode {selected_modes[0]}')
ax.set_ylabel(f'V-mode {selected_modes[1]}')
ax.set_zlabel(f'V-mode {selected_modes[2]}')
ax.legend()
plt.show()
```
After performing and analyzing the SVD, we can build a linear classifier using linear discriminant analysis (LDA).
We'll use the discriminant_analysis.LinearDiscriminantAnalysis package from sklearn. To ensure our classifier is working properly, 
we can build a linear classifier (LDA) that can reasonable identify/classify/differentiate between just two digits initially: 
```
# Filter the dataset to keep only the digits 4 and 5
selected_digits = ['4', '5']
mask = np.isin(y, selected_digits)
X_filtered, y_filtered = X[mask], y[mask]

# Reshape images into column vectors
X_flattened = X_filtered.reshape(X_filtered.shape[0], -1)

# Standardize the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_flattened)

# Split the dataset into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y_filtered, test_size=0.2, random_state=42)

# Perform LDA and fit the classifier
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
```
After performing LDA, we can evaluate the classifier on our untouched test set: 
```
# Evaluate the classifier on the test set
y_pred = lda.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(cm)

accuracy_4 = cm[0, 0] / cm[0].sum()
accuracy_5 = cm[1, 1] / cm[1].sum()

print(f"Accuracy for digit 4: {accuracy_4:.2f}")
print(f"Accuracy for digit 5: {accuracy_5:.2f}")
```
This yields the following 
```
Classification Report:
              precision    recall  f1-score   support

           4       0.99      0.99      0.99      1342
           5       0.98      0.99      0.99      1286

    accuracy                           0.99      2628
   macro avg       0.99      0.99      0.99      2628
weighted avg       0.99      0.99      0.99      2628

Confusion Matrix:
[[1322   20]
 [  14 1272]]
Accuracy for digit 4: 0.99
Accuracy for digit 5: 0.99
```
Then, we can expand our LDA to 3 different digits. Arbitrarily, let's choose 0, 1, 2: 
```
# Filter the dataset to keep only the digits 1, 4, and 7
selected_digits = ['0', '1', '2']
mask = np.isin(y, selected_digits)
X_filtered, y_filtered = X[mask], y[mask]

# Reshape images into column vectors
X_flattened = X_filtered.reshape(X_filtered.shape[0], -1)

# Standardize the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_flattened)

# Split the dataset into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y_filtered, test_size=0.2, random_state=42)

# Perform LDA and fit the classifier
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = lda.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(cm)

accuracy_0 = cm[0, 0] / cm[0].sum()
accuracy_1 = cm[1, 1] / cm[1].sum()
accuracy_2 = cm[2, 2] / cm[2].sum()

print(f"Accuracy for digit 0: {accuracy_0:.2f}")
print(f"Accuracy for digit 1: {accuracy_1:.2f}")
print(f"Accuracy for digit 2: {accuracy_2:.2f}")
```
This yields: 
```
Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.98      0.98      1386
           1       1.00      0.74      0.85      1592
           2       0.75      0.98      0.85      1376

    accuracy                           0.89      4354
   macro avg       0.91      0.90      0.89      4354
weighted avg       0.91      0.89      0.89      4354

Confusion Matrix:
[[1355    0   31]
 [   4 1173  415]
 [  28    5 1343]]
Accuracy for digit 0: 0.98
Accuracy for digit 1: 0.74
Accuracy for digit 2: 0.98
```


