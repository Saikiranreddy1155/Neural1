700763320 saikiran reddy chekkabandi
1.
To run the code, you will need the following Python package:

TensorFlow: To install TensorFlow, run the following command:
bash
Copy
pip install tensorflow
Overview of Tensor Operations
1. Creating a Random Tensor
The code begins by creating a random tensor of shape (4, 6) using tf.random.uniform(). This function generates a tensor with random values sampled from a uniform distribution between 0 and 1.

python
Copy
tensor = tf.random.uniform((4, 6), dtype=tf.float32)
The resulting tensor has a shape of (4, 6) and data type tf.float32.
2. Getting the Rank and Shape of the Tensor
Next, we compute the rank and shape of the tensor:

python
Copy
tensor_rank = tf.rank(tensor)
tensor_shape = tensor.shape
Rank refers to the number of dimensions in the tensor (i.e., how many axes it has). In this case, the rank is 2 since the tensor has two dimensions: (4, 6).
Shape refers to the size of the tensor in each dimension. The shape of our tensor is (4, 6).
3. Reshaping the Tensor
We reshape the tensor to a new shape of (2, 3, 4) using tf.reshape(). This operation preserves the total number of elements but rearranges them into a new shape:

python
Copy
reshaped_tensor = tf.reshape(tensor, (2, 3, 4))
The new shape (2, 3, 4) means the tensor now has three dimensions.
Note: The total number of elements remains the same: 4 * 6 = 2 * 3 * 4 = 24 elements.
4. Transposing the Tensor
The reshaped tensor is then transposed to swap the first and second dimensions. This is done using tf.transpose():

python
Copy
transposed_tensor = tf.transpose(reshaped_tensor, perm=[1, 0, 2])
The perm=[1, 0, 2] argument specifies that the second dimension (size 3) should come first, followed by the first dimension (size 2), and then the third dimension (size 4).
The result is a tensor of shape (3, 2, 4).
5. Creating a Smaller Tensor
We then create a smaller tensor of shape (1, 4) using tf.random.uniform():

python
Copy
smaller_tensor = tf.random.uniform((1, 4), dtype=tf.float32)
The shape (1, 4) means it has 1 row and 4 columns.
6. Broadcasting and Adding Tensors
Finally, we perform broadcasting by adding the smaller_tensor to the transposed_tensor. Broadcasting allows tensors of different shapes to be added together as long as their shapes are compatible:

python
Copy
broadcasted_tensor = smaller_tensor + transposed_tensor
The smaller tensor ((1, 4)) is broadcasted across the first dimension of the transposed_tensor ((3, 2, 4)), meaning it is replicated for each of the 3 "rows" in the larger tensor.
The result is a tensor of shape (3, 2, 4).
Output
The output will display the following:

The original random tensor.
The rank and shape of the original tensor.
The reshaped tensor of shape (2, 3, 4).
The transposed tensor of shape (3, 2, 4).
The smaller tensor of shape (1, 4).
The result after broadcasting and adding the tensors.
Example Code
python
Copy
import tensorflow as tf

# Creating a random tensor of shape (4, 6)
tensor = tf.random.uniform((4, 6), dtype=tf.float32)
print("Random Tensor:")
print(tensor)

tensor_rank = tf.rank(tensor)
tensor_shape = tensor.shape

print(f"Rank: {tensor_rank}")
print(f"Shape: {tensor_shape}")

# Reshaping the tensor to (2, 3, 4)
reshaped_tensor = tf.reshape(tensor, (2, 3, 4))
print("Reshaped Tensor (2, 3, 4):")
print(reshaped_tensor)

# Transposing the tensor to (3, 2, 4)
transposed_tensor = tf.transpose(reshaped_tensor, perm=[1, 0, 2])
print("Transposed Tensor (3, 2, 4):")
print(transposed_tensor)

# Creating a smaller tensor of shape (1, 4)
smaller_tensor = tf.random.uniform((1, 4), dtype=tf.float32)
print("Smaller Tensor (1, 4):")
print(smaller_tensor)

# Broadcasting the smaller tensor and adding it to the larger tensor
broadcasted_tensor = smaller_tensor + transposed_tensor
print("Result after Broadcasting and Adding:")
print(broadcasted_tensor)





2.
   1. Define True Values and Model Predictions
Regression Task:

y_true_regression is the array of true values for the regression task.
y_pred_regression contains the predicted values from the model.
Classification Task (One-hot encoded):

y_true_classification is the one-hot encoded array of true values for a classification task with 3 classes.
y_pred_classification contains the predicted probabilities for each of the 3 classes.
2. Compute Loss Functions
Mean Squared Error (MSE) for the regression task is computed using TensorFlow's tf.keras.losses.MeanSquaredError():
python
Copy
mse_loss = tf.keras.losses.MeanSquaredError()
mse_value = mse_loss(y_true_regression, y_pred_regression)
MSE measures the average of the squared differences between true and predicted values.
Categorical Cross-Entropy (CCE) for the classification task is computed using tf.keras.losses.CategoricalCrossentropy():
python
Copy
cce_loss = tf.keras.losses.CategoricalCrossentropy()
cce_value = cce_loss(y_true_classification, y_pred_classification)
CCE measures the difference between the true one-hot encoded class and the predicted probabilities.
3. Modify Predictions
Small random perturbations are added to the predictions for both tasks to simulate a change in predictions:
python
Copy
y_pred_regression_modified = y_pred_regression + np.random.normal(0, 0.1, y_pred_regression.shape)
y_pred_classification_modified = y_pred_classification + np.random.normal(0, 0.1, y_pred_classification.shape)
4. Compute New Loss Values
The new loss values are computed after modifying the predictions:
python
Copy
mse_modified_value = mse_loss(y_true_regression, y_pred_regression_modified)
cce_modified_value = cce_loss(y_true_classification, y_pred_classification_modified)
5. Visualize the Loss Values
The code generates loss curves by slightly perturbing the predictions over a range of values from -1 to 1.

The loss values for both MSE and CCE are recorded and plotted:

MSE Loss is calculated for perturbations applied to the regression task predictions.
CCE Loss is calculated for perturbations applied to the classification task predictions.
Plotting the Results: The loss values are plotted for both tasks using matplotlib. The first subplot displays the MSE Loss curve, and the second subplot shows the CCE Loss curve.

6. Final Output
The loss values are printed:
Mean Squared Error (MSE) and Categorical Cross-Entropy (CCE) values for the original and modified predictions.
The loss curves will show how the loss changes as the predictions are perturbed.
Code Example
python
Copy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define true values and model predictions for a regression task
y_true_regression = np.array([3.0, 2.5, 4.0, 5.1, 3.8])
y_pred_regression = np.array([2.9, 2.7, 4.2, 5.0, 3.5])

# Define true values and model predictions for a classification task (one-hot encoded)
y_true_classification = np.array([[0, 1, 0],  # Class 1
                                  [0, 0, 1],  # Class 2
                                  [0, 1, 0],  # Class 1
                                  [1, 0, 0],  # Class 0
                                  [0, 0, 1]]) # Class 2

y_pred_classification = np.array([[0.1, 0.8, 0.1],  # Pred for Class 1
                                  [0.2, 0.2, 0.6],  # Pred for Class 2
                                  [0.3, 0.6, 0.1],  # Pred for Class 1
                                  [0.7, 0.2, 0.1],  # Pred for Class 0
                                  [0.1, 0.1, 0.8]]) # Pred for Class 2

# Compute Mean Squared Error (MSE) for regression
mse_loss = tf.keras.losses.MeanSquaredError()
mse_value = mse_loss(y_true_regression, y_pred_regression)

# Compute Categorical Cross-Entropy (CCE) for classification
cce_loss = tf.keras.losses.CategoricalCrossentropy()
cce_value = cce_loss(y_true_classification, y_pred_classification)

print(f"Mean Squared Error (MSE): {mse_value.numpy()}")
print(f"Categorical Cross-Entropy (CCE): {cce_value.numpy()}")

# Modify predictions slightly for both regression and classification
y_pred_regression_modified = y_pred_regression + np.random.normal(0, 0.1, y_pred_regression.shape)
y_pred_classification_modified = y_pred_classification + np.random.normal(0, 0.1, y_pred_classification.shape)

# Compute new loss values
mse_modified_value = mse_loss(y_true_regression, y_pred_regression_modified)
cce_modified_value = cce_loss(y_true_classification, y_pred_classification_modified)

print(f"Modified Mean Squared Error (MSE): {mse_modified_value.numpy()}")
print(f"Modified Categorical Cross-Entropy (CCE): {cce_modified_value.numpy()}")

# Plotting the loss values
loss_values_mse = []
loss_values_cce = []

# Create a range of perturbations to the predictions
perturbations = np.linspace(-1, 1, 100)

for perturb in perturbations:
    # Modify predictions slightly for the regression task
    y_pred_regression_perturbed = y_pred_regression + perturb
    mse_loss_value = mse_loss(y_true_regression, y_pred_regression_perturbed)
    loss_values_mse.append(mse_loss_value.numpy())

    # Modify predictions slightly for the classification task
    y_pred_classification_perturbed = y_pred_classification + perturb
    cce_loss_value = cce_loss(y_true_classification, y_pred_classification_perturbed)
    loss_values_cce.append(cce_loss_value.numpy())

# Plot the results
plt.figure(figsize=(12, 6))

# MSE Plot
plt.subplot(1, 2, 1)
plt.plot(perturbations, loss_values_mse, label="MSE Loss")
plt.title("Mean Squared Error (MSE) Loss")
plt.xlabel("Perturbation")
plt.ylabel("Loss")

# CCE Plot
plt.subplot(1, 2, 2)
plt.plot(perturbations, loss_values_cce, label="CCE Loss", color='red')
plt.title("Categorical Cross-Entropy (CCE) Loss")
plt.xlabel("Perturbation")
plt.ylabel("Loss")

plt.tight_layout()
plt.show()
Final Result:
Printed Loss Values: You'll see the MSE and CCE loss values for the original and perturbed predictions.
Loss Curves: The plots will show how the loss changes when the predictions are perturbed, giving insights into how each type of loss behaves with changes in the predictions.



3.
Steps in the Code:
Loading the MNIST Dataset:

The MNIST dataset is loaded using TensorFlow’s tf.keras.datasets.mnist.load_data() function. It contains 28x28 pixel images of handwritten digits (0-9) and their corresponding labels.
x_train, y_train contain the training data (images and labels), and x_val, y_val contain the validation data.
Data Normalization:

The images are normalized to a range between 0 and 1 by dividing by 255.0:
python
Copy
x_train, x_val = x_train / 255.0, x_val / 255.0
Reshaping Data:

The images are reshaped to have a shape of (28, 28, 1) to be compatible with the CNN model's input. This adds a channel dimension to the grayscale images:
python
Copy
x_train = x_train.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)
Model Creation (CNN Architecture):

The function create_model() defines a CNN model with the following architecture:
Conv2D Layer (32 filters, 3x3 kernel) followed by MaxPooling2D.
Conv2D Layer (64 filters, 3x3 kernel) followed by MaxPooling2D.
Flatten Layer to convert the 2D feature maps to 1D vectors.
Dense Layer (64 units) with ReLU activation.
Dense Layer (10 units) with Softmax activation for multi-class classification (digits 0-9).
The model is compiled with a loss function of sparse_categorical_crossentropy and metrics accuracy:

python
Copy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
Training the Models:

The model is trained for 5 epochs using Adam optimizer first, and the training and validation history are saved to adam_history.
python
Copy
adam_history = adam_model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val), verbose=2)
Next, the model is compiled with SGD optimizer and trained again for 5 epochs, saving the history to sgd_history:
python
Copy
sgd_model.compile(optimizer=tf.keras.optimizers.SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
sgd_history = sgd_model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val), verbose=2)
Plotting the Results:

Finally, the code plots the training and validation accuracy for both optimizers (Adam and SGD) across the 5 epochs.
The first subplot shows the accuracy for the Adam optimizer.
The second subplot shows the accuracy for the SGD optimizer.
Code Explanation:
python
Copy
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

# Normalize the images to the range [0, 1]
x_train, x_val = x_train / 255.0, x_val / 255.0

# Reshape data to be compatible with the model input (28x28 pixels, 1 channel)
x_train = x_train.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',  # placeholder for optimizer, will change later
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Create models
adam_model = create_model()
sgd_model = create_model()

# Train with Adam optimizer
adam_history = adam_model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val), verbose=2)

# Train with SGD optimizer
sgd_model.compile(optimizer=tf.keras.optimizers.SGD(),  # Use SGD optimizer
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
sgd_history = sgd_model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val), verbose=2)

# Plot training and validation accuracy for both optimizers
plt.figure(figsize=(12, 6))

# Adam model accuracy
plt.subplot(1, 2, 1)
plt.plot(adam_history.history['accuracy'], label='Training Accuracy (Adam)', color='blue')
plt.plot(adam_history.history['val_accuracy'], label='Validation Accuracy (Adam)', color='orange')
plt.title('Training and Validation Accuracy (Adam)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# SGD model accuracy
plt.subplot(1, 2, 2)
plt.plot(sgd_history.history['accuracy'], label='Training Accuracy (SGD)', color='blue')
plt.plot(sgd_history.history['val_accuracy'], label='Validation Accuracy (SGD)', color='orange')
plt.title('Training and Validation Accuracy (SGD)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
Output:
Training and Validation Accuracy Plots:
The left plot shows how the training and validation accuracy of the model evolves over the 5 epochs when trained with the Adam optimizer.
The right plot shows how the training and validation accuracy evolve over the 5 epochs when trained with the SGD optimizer.
Insights:
Adam Optimizer: Typically, the Adam optimizer converges faster and often performs better in practice, especially on complex tasks like image classification.
SGD Optimizer: While SGD is simpler and more computationally efficient, it might take longer to converge and may not perform as well as Adam, especially when learning rates are not tuned properly.
You can observe the difference in accuracy between both optimizers, giving insights into their performance on the MNIST dataset.




4.
To run this project, you'll need the following Python packages:

tensorflow
numpy
matplotlib (optional for visualizations)
You can install the required dependencies by running:

bash
Copy
pip install tensorflow numpy matplotlib
Dataset
The MNIST dataset is a collection of 70,000 handwritten digits (28x28 pixels each) labeled from 0 to 9. The dataset is split into:

60,000 images for training
10,000 images for validation
The neural network will be trained to predict the digits based on these images.

Code Explanation
Loading and Preprocessing the Data:

The mnist.load_data() function is used to load the training and validation data.
The pixel values of the images are normalized to a range between 0 and 1 by dividing the values by 255. This helps with faster and more stable training.
The images are reshaped to have a single channel (grayscale) in the shape (28, 28, 1) to match the expected input format for the neural network.
The labels are one-hot encoded to represent each class (digit 0-9) as a 10-dimensional vector.
Building the Neural Network Model:

The model consists of three layers:

Flatten: This layer flattens the 28x28 image into a 1D vector of 784 pixels.
Dense Layer: A fully connected layer with 128 neurons and ReLU activation. This layer learns the patterns in the data.
Dropout Layer: A regularization technique to prevent overfitting by randomly setting some of the neurons to zero during training.
Output Layer: A softmax layer with 10 units, each corresponding to one of the 10 possible digits (0-9). The softmax function outputs probabilities for each class.
Compiling the Model:

The model is compiled with the Adam optimizer, categorical crossentropy as the loss function, and accuracy as the evaluation metric.
Training with TensorBoard Logging:

The model is trained for 5 epochs on the training data and validated on the validation data.
During training, the TensorBoard callback is used to log various metrics (e.g., accuracy and loss) for visualization.
You can monitor training in real-time by using TensorBoard.
Running TensorBoard
After running the training process, you can visualize the training and validation metrics in real-time using TensorBoard. This helps you monitor the progress of the model, check for overfitting, and understand how the model is learning.

In Jupyter Notebook:
If you're using Jupyter Notebook, run the following commands after training:

python
Copy
%load_ext tensorboard
%tensorboard --logdir logs/fit
In Command Line:
If you're running the script outside of Jupyter Notebook, start TensorBoard from the command line:

bash
Copy
tensorboard --logdir=logs/fit
Then, open the URL http://localhost:6006/ in your web browser to view the training and validation metrics.

Training Progress
Once TensorBoard is running, you'll be able to see visualizations like:

Loss curves for both training and validation data.
Accuracy curves to track how well the model is performing during training.
Histograms of activations and weights after each epoch.
Conclusion
This project provides a simple demonstration of using a neural network to classify handwritten digits. The use of TensorBoard helps in visualizing the training process and tracking performance metrics in real-time.










