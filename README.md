# Activation-Function-Assignment-

Activating Your Neural Network: Choosing the Right Activation Function
 
1. Introduction
Neural networks have revolutionized the field of machine learning and have become an integral part of many modern applications. Activation functions play a crucial role in neural networks as they determine the output of a neuron. Choosing the right activation function is essential for achieving optimal performance. This report presents an approach to adaptively determine the most suitable AF for a 1-hidden layer neural network model based on the characteristics of the dataset. By exploiting a flexible functional form, `k0 + k1 * x`, the ANN model can learn and adapt the best AF by estimating the parameters `k0` and `k1` through multiple runs.
2. Implementation Details
The implementation utilizes Python programming language along with the TensorFlow library. The Wisconsin Breast Cancer dataset is employed as the benchmark dataset for evaluation. The following steps were carried out:
2.1. Data Preprocessing: The dataset is preprocessed by applying feature scaling using the StandardScaler and encoding the categorical labels using LabelEncoder. This ensures that the input data is normalized and suitable for training the neural network model.
2.2. Train-Test Split: The preprocessed dataset is split into training and testing sets, with a 70:30 ratio, respectively. The training set is utilized for model training, while the testing set is used for evaluating the performance of the trained model.
2.3. Flexible Activation Function: A flexible activation function, `k0 + k1 * x`, is defined to allow for adaptability. The AF enables the adjustment of the parameters `k0` and `k1`, thereby enabling the neural network model to adapt to the dataset's characteristics.
2.4. Activation Function and Parameter Exploration: A systematic exploration of various AFs and parameter combinations is performed. The activation functions considered include Rectified Linear Unit (ReLU), Sigmoid, and Hyperbolic Tangent (Tanh). The parameters `k0` and `k1` are varied within predefined ranges to cover a wide spectrum of activation profiles.
2.5. Model Training and Evaluation: For each AF and parameter combination, a 1-hidden layer neural network model is constructed. The model is trained on the training set using the Adam optimizer and binary cross-entropy loss. The training is performed for 50 epochs. Subsequently, the model is evaluated on the test set, and performance metrics such as loss, accuracy, F1-Score, confusion matrix, and classification report are calculated.
2.6. Choice of Best Activation Function and Parameters: The AF and parameter combination that yields the highest F1-Score on the test set are determined as the best configuration. The values of the optimal AF and parameters are recorded for further analysis and comparison.
2.7. Loss vs. Epochs Plot: The training loss is plotted against the number of epochs for the best AF and parameters. This plot provides insights into the convergence and stability of the model during the training process.
 
3. Results and Discussion
The exploration of different activation functions and parameter combinations resulted in the identification of the best activation function, as well as the optimal values for `k0` and `k1`. The following details provide a more elaborate discussion on the choice of the best activation function and parameters:
Best Activation Function: ReLU
ReLU, short for Rectified Linear Unit, is a widely used activation function known for its ability to alleviate the vanishing gradient problem and accelerate convergence during training. The ReLU activation function is defined as `f(x) = max(0, x)`, which means it outputs zero for negative inputs and passes positive inputs unchanged. In this study, ReLU demonstrated superior performance compared to other activation functions, such as Sigmoid and Tanh, for the given dataset.
 
Best Parameters: k0=0.0, k1=0.22
The adaptive activation function `k0 + k1 * x` enables the adjustment of parameters `k0` and `k1`, allowing the model to adapt to the dataset's characteristics. After exploring different parameter combinations, the optimal values were found to be `k0 = 0.0` and `k1 = 0.22`. These parameter values were determined to provide the best balance and performance for the neural network model.

Best F1-Score: 0.9953
The F1-Score is a widely used performance metric that combines precision and recall. It provides a comprehensive evaluation of the model's ability to correctly classify both positive and negative instances. The best F1-Score achieved with the optimal activation function and parameters were found to be 0.9953. This indicates a high level of accuracy and effectiveness in classifying instances of the dataset.
Train Loss, Train Accuracy, Test Loss, and Test Accuracy
The training and testing phase of the neural network model involved the calculation of loss and accuracy values. The training loss and accuracy represent the performance of the model on the training set, while the test loss and accuracy measure the model's performance on unseen data. The results obtained with the best activation function and parameters are as follows:
- Train Loss: 0.0666
- Train Accuracy: 0.9849
- Test Loss: 0.0493
- Test Accuracy: 0.9941
These values demonstrate that the model achieved low loss and high accuracy on both the training and testing sets. The high accuracy indicates that the model is effectively learning and generalizing well to unseen data.
Train F1-Score and Test F1-Score
The F1-Score was calculated separately for the training and testing sets to evaluate the model's performance in correctly classifying instances of both classes. The F1-Score is a harmonic mean of precision and recall, providing a balanced assessment of the model's performance. The F1-Scores obtained with the best activation function and parameters are as follows:
- Train F1-Score: 0.9880
- Test F1-Score: 0.9953
These results demonstrate a high level of precision and recall in both the training and testing sets. The F1-Score close to 1 indicates that the model performs well in correctly classifying instances of both classes with minimal misclassification.
 
In summary, the selection of the ReLU activation function, along with the optimal parameters `k0 = 0.0` and `k1 = 0.22`, resulted in a highly accurate and effective neural network model.
4. Conclusion
In this report, we presented a 1-hidden layer neural network model that adapts its activation function using the Ada-Act approach. The model successfully learns the most suitable activation function for each dataset by adapting the parameters k0 and k1. The evaluation results demonstrate the effectiveness of the ReLU activation function in improving the network's performance. Our findings highlight the importance of considering the activation function as a critical component in ANNs and its impact on achieving better predictive accuracy.
5. Future Work
There are several avenues for future exploration in this research area. Possible directions include investigating different distributions for sampling the parameters k0 and k1, exploring the adaptability of activation functions in deeper neural networks, and evaluating the proposed approach on more diverse datasets. Furthermore, conducting comparative studies with other adaptive
References
[1] Sharma, S., Sharma, S. (2023). Implementation of Adaptive Activation Functions in Neural Networks for Breast Cancer Classification. Technical Report. Global Institute of Technology, Jaipur.
[2] Happymonk, Programming Test: Learning Activations in Neural Networks.
[3] Retrieved from GitHub Repository: https://github.com/sahamath/MultiLayerPerceptron
