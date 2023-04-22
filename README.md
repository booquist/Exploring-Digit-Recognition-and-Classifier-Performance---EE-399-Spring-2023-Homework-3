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

```python
# Load MNIST dataset
mnist = fetch_openml('mnist_784')
X, y = mnist['data'].to_numpy(), mnist['target']
```

The images are then reshaped into column vectors and standardized to have a mean of zero and a standard deviation of one.
```python
# Reshape images into column vectors
X_flattened = X.reshape(X.shape[0], -1)

# Standardize the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_flattened)
```

**Performing Singular Value Decomposition (SVD)** <br>
Next, we perform SVD on the standardized data using NumPy's linalg.svd method. 
We keep the first 50 dimensions (or modes) for analysis and plot the 9 most important SVD modes as "eigen-digits".

```python
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
```python
# Calculate the number of modes required for good image reconstruction
energy = np.sum(s**2)
cumulative_energy = np.cumsum(s**2)
r = np.argmax(cumulative_energy / energy > 0.9)
```

Next, we project the standardized data onto the selected V-modes, which in this case are modes 1, 2, and 4.
```python
# Project data onto selected V-modes
selected_modes = [1, 2, 4]  # Using 1-based indexing
X_projected = X_standardized @ Vt[([mode - 1 for mode in selected_modes]), :].T
```

We also create a 3D scatter plot using the projected data to visualize the distribution of the digits in the reduced feature space.
```python
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
```python
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
```python
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
```python
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
Now that we can confirm our LDA works reasonably well, we can completely expand it to try and determine the easiest and most difficult digits to separate. 
We'll discuss the results later: 
```python
digit_pairs = list(itertools.combinations(range(10), 2))
accuracies = []

for digit_pair in digit_pairs:
    selected_digits = [str(d) for d in digit_pair]
    mask = np.isin(y, selected_digits)
    X_filtered, y_filtered = X[mask], y[mask]

    X_flattened = X_filtered.reshape(X_filtered.shape[0], -1)
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X_flattened)

    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y_filtered, test_size=0.2, random_state=42)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    y_pred = lda.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    accuracy_1 = cm[0, 0] / cm[0].sum()
    accuracy_2 = cm[1, 1] / cm[1].sum()
    avg_accuracy = (accuracy_1 + accuracy_2) / 2

    accuracies.append(avg_accuracy)
    print(f"Accuracy for digit pair {digit_pair}: {avg_accuracy:.2f}")

min_index = np.argmin(accuracies)
max_index = np.argmax(accuracies)

print(f"Most difficult pair to separate: {digit_pairs[min_index]}, accuracy: {accuracies[min_index]:.2f}")
print(f"Easiest pair to separate: {digit_pairs[max_index]}, accuracy: {accuracies[max_index]:.2f}")
```
LDA's and linear classifiers are great, but lets use slightly more sophisticated algorithms. We can develop classifiers using SVM (Support Vector Machines), and Decision Tree Classifiers to test and determine how well they perform: 
```python
# Split the dataset into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the SVM classifier
svm = SVC(kernel='rbf', C=1, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM classifier accuracy: {accuracy_svm:.2f}")
```
```python
# Train and evaluate the Decision Tree classifier
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)
y_pred_dtree = dtree.predict(X_test)
accuracy_dtree = accuracy_score(y_test, y_pred_dtree)
print(f"Decision Tree classifier accuracy: {accuracy_dtree:.2f}")
```

We can face use our SVM and decision tree to separate digits, and compare these results to our LDA: 
```python 
def evaluate_classifier_accuracy(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    accuracies = [cm[i, i] / cm[i].sum() for i in range(cm.shape[0])]
    avg_accuracy = np.mean(accuracies)
    return avg_accuracy

digit_pairs = list(itertools.combinations(range(10), 2))

svm_accuracies = []
dtree_accuracies = []

for digit_pair in digit_pairs:
    selected_digits = [str(d) for d in digit_pair]
    mask = np.isin(y, selected_digits)
    X_filtered, y_filtered = X[mask], y[mask]

    X_flattened = X_filtered.reshape(X_filtered.shape[0], -1)
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X_flattened)

    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y_filtered, test_size=0.2, random_state=42)

    svm = SVC(kernel='rbf', C=1, random_state=42)
    svm_accuracy = evaluate_classifier_accuracy(svm, X_train, y_train, X_test, y_test)
    svm_accuracies.append(svm_accuracy)

    dtree = DecisionTreeClassifier(random_state=42)
    dtree_accuracy = evaluate_classifier_accuracy(dtree, X_train, y_train, X_test, y_test)
    dtree_accuracies.append(dtree_accuracy)

    print(f"Accuracy for digit pair {digit_pair} - SVM: {svm_accuracy:.2f}, Decision Tree: {dtree_accuracy:.2f}")

svm_min_index = np.argmin(svm_accuracies)
svm_max_index = np.argmax(svm_accuracies)

dtree_min_index = np.argmin(dtree_accuracies)
dtree_max_index = np.argmax(dtree_accuracies)

print(f"SVM - Most difficult pair to separate: {digit_pairs[svm_min_index]}, accuracy: {svm_accuracies[svm_min_index]:.2f}")
print(f"SVM - Easiest pair to separate: {digit_pairs[svm_max_index]}, accuracy: {svm_accuracies[svm_max_index]:.2f}")

print(f"Decision Tree - Most difficult pair to separate: {digit_pairs[dtree_min_index]}, accuracy: {dtree_accuracies[dtree_min_index]:.2f}")
print(f"Decision Tree - Easiest pair to separate: {digit_pairs[dtree_max_index]}, accuracy: {dtree_accuracies[dtree_max_index]:.2f}")
```
We'll discuss the results of this separation in the following section. 

## IV. Computational Results
**Singular Value Decomposition (SVD)** <br>
We performed SVD on the standardized dataset to identify the most important directions in the MNIST dataset. Figure 1 shows the first 9 SVD modes, also known as "eigen-digits".

![image](https://user-images.githubusercontent.com/103399658/233752376-ca728063-4ffb-4e0e-90ff-d026ef6a1a96.png)
<br>*Figure 1: First 9 SVD modes (eigen-digits)*<br>
The eigen-digits reveal the underlying structure of the data, highlighting the similarities and differences between digits. For example, SVD Mode 5 appears to capture the "loop" structure of digits like "6" and "9", while SVD Mode 9 captures the "loop" structure of "8". <br>

We also calculated the number of modes required to retain at least 90% of the energy in the dataset, which was found to be 237. We projected the standardized data onto the first 3 SVD modes and created a 3D scatter plot to visualize the distribution of digits in the reduced feature space. Figure 2 shows the resulting plot: 

![image](https://user-images.githubusercontent.com/103399658/233752724-1cd2ffb1-cca5-4f27-9190-aaf0836f4476.png)
<br>*Figure 2: 3D scatter plot of MNIST digits projected onto the first 3 SVD modes (eigen-digits)*<br>

The plot reveals that different digits *(mostly)* occupy different regions of the reduced feature space, which suggests that they may be separable using machine learning techniques. <br>

Moving on to our LDA, we can finally analyze the results of our three-digit separation, and determine which digits are the most difficult to separate. 
Starting off with the three-digit separation in the most basic linear classifier, we find: 
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
Looking at the results, we see that the classifier is doing well for digits 0 and 2, with accuracies of 0.98 and 0.98, respectively. However, for digit 1, the accuracy is only 0.74. This suggests that the classifier is having more difficulty distinguishing between 1s and other digits. The overall accuracy of the model is 0.89, which is decent but could be improved. <br>
Moving on to analyzing which two digits are the most difficult and easiest to separate, we find the following results: 
```
Accuracy for digit pair (0, 1): 0.99
Accuracy for digit pair (0, 2): 0.99
Accuracy for digit pair (0, 3): 0.99
...
Accuracy for digit pair (8, 9): 0.97
Most difficult pair to separate: (2, 6), accuracy: 0.75
Easiest pair to separate: (6, 7), accuracy: 1.00
```

The results show that the model has high accuracy for most digit pairs, with accuracy ranging from 0.97 to 1.00. The most difficult digit pair to separate is (2,6) with an accuracy of 0.75, while the easiest digit pair to separate is (6,7) with an accuracy of 1.00. <br> <br>
The pair (2, 6) may be difficult to separate because they share similar features, such as loops and curves in their shapes. Specifically, both digits have a loop at the top and a curve at the bottom, making them visually similar. Additionally, the way the loops and curves are shaped may vary depending on the handwriting style, further complicating their separation. This similarity in appearance can make it difficult for a classifier to accurately distinguish between them, resulting in a lower accuracy for this digit pair. <br>

Of course, SVM and decision trees are a much more modern way of doing machine learning, so let's take a look at the results of both: <br>
**SVM**
```python
# Split the dataset into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the SVM classifier
svm = SVC(kernel='rbf', C=1, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM classifier accuracy: {accuracy_svm:.2f}")
```
yields "SVM classifier accuracy: 0.98". <br>

**Decision Tree**
```python
# Train and evaluate the Decision Tree classifier
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)
y_pred_dtree = dtree.predict(X_test)
accuracy_dtree = accuracy_score(y_test, y_pred_dtree)
print(f"Decision Tree classifier accuracy: {accuracy_dtree:.2f}")
``` 
yields "Decision Tree classifier accuracy: 0.87" <br>

As expected, our SVM has a high accuracy of 98% on the test data. Surprisingly, the decision tree only has an accuracy of 87%. However, as we'll see later, 
it ends up being more generalizable. <br>

Repeating the process of finding the most easily and difficult to separate for SVM and decision trees, we find: 
```
Accuracy for digit pair (0, 1) - SVM: 0.99, Decision Tree: 1.00
Accuracy for digit pair (0, 2) - SVM: 0.99, Decision Tree: 0.98
...
Accuracy for digit pair (8, 9) - SVM: 0.99, Decision Tree: 0.97
SVM - Most difficult pair to separate: (7, 9), accuracy: 0.97
SVM - Easiest pair to separate: (3, 6), accuracy: 1.00
Decision Tree - Most difficult pair to separate: (2, 3), accuracy: 0.95
Decision Tree - Easiest pair to separate: (0, 1), accuracy: 1.00
```
Unsurprisingly, both the SVM and decision tree had an easier time separating digits than the LDA, with (2, 3) proving the most difficult for the decision tree. This may because both digits contain a similar hook at the top. <br> 

I attempted to use the SVM and decision tree models to classify my own handwriting: 

![image](https://user-images.githubusercontent.com/103399658/233753822-6fae168f-2700-4c7e-9277-9cb03e1ef783.png)
![image](https://user-images.githubusercontent.com/103399658/233753836-0640e4f5-b719-441e-ac0f-da0c97ea885f.png)

I implemented this classification using the following code: <br>
**SVM**
```python
from PIL import Image

# Open the image and convert it to grayscale
image = Image.open("MNIST-test.png").convert("L")

# Resize the image to 28x28 pixels
resized_image = image.resize((28, 28))

# Flatten the image to a 1D array
flattened_image = np.array(resized_image).flatten()

# Scale the image using the same scaler used for the training data
scaled_image = scaler.transform([flattened_image])

# Predict the digit using the SVM
digit = svm.predict(scaled_image)[0]
print(f"Predicted digit: {digit}")
```
**Decision Tree**
```python 
# Open the image and convert it to grayscale
image = Image.open("MNIST-test.png").convert("L")

# Resize the image to 28x28 pixels
resized_image = image.resize((28, 28))

# Flatten the image to a 1D array
flattened_image = np.array(resized_image).flatten()

# Scale the image using the same scaler used for the training data
scaled_image = scaler.transform([flattened_image])

# Predict the digit using the Decision Tree classifier
digit = dtree.predict(scaled_image)[0]
print(f"Predicted digit (image 1) with Decision Tree: {digit}")

# Open the image and convert it to grayscale
image2 = Image.open("MNIST-test-other.png").convert("L")

# Resize the image to 28x28 pixels
resized_image2 = image2.resize((28, 28))

# Flatten the image to a 1D array
flattened_image2 = np.array(resized_image2).flatten()

# Scale the image using the same scaler used for the training data
scaled_image2 = scaler.transform([flattened_image2])

# Predict the digit using the Decision Tree classifier
digit2 = dtree.predict(scaled_image2)[0]
print(f"Predicted digit (image 2) with Decision Tree: {digit2}")
```
Interestingly, I found that the SVM incorrectly classified both the 1 and 2 as 5. However, the decision tree correctly identified both digits. This may be due to severe overfitting on part of the SVM.

