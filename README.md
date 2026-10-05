# Fashion-MNIST Classification: Logistic Regression, ANN, and CNN

This notebook presents an end-to-end comparison of three image-classification approaches on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset:

- **Logistic Regression** provides a fast, interpretable linear baseline.
- **Artificial Neural Network (ANN)** learns nonlinear relationships from flattened pixel vectors.
- **Convolutional Neural Network (CNN)** uses the spatial structure of each image to learn local visual features.

The aim is not only to find the most accurate model, but also to show how model capacity and image-aware architecture affect classification performance.

## Dataset

Fashion-MNIST contains grayscale images of clothing and accessories. Each image is $28 \times 28$ pixels and belongs to one of 10 classes:

| Label | Class |
| ---: | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

Keras provides 60,000 training images and 10,000 test images. The notebook further divides the original training data into:

- **50,000 training examples**
- **10,000 validation examples**
- **10,000 held-out test examples**

The validation split is stratified with a fixed random state so that class proportions remain consistent and results are reproducible.

## What the Notebook Does

### 1. Sets up a reproducible environment

The notebook imports NumPy, pandas, Matplotlib, scikit-learn, and Keras. It sets the Keras random seed to `42` and reports the installed Keras version and backend.

### 2. Loads and explores the data

The exploratory analysis includes:

- Inspecting image and label shapes.
- Displaying individual images and a grid of 20 samples.
- Mapping numeric labels to readable class names.
- Checking data types, pixel ranges, missing values, and infinite values.
- Checking for exact duplicate image vectors.
- Reviewing class frequencies with a table and bar chart.

### 3. Preprocesses the images

Pixel values are converted from integers in the range `[0, 255]` to `float32` values in `[0.0, 1.0]` by dividing by `255.0`.

Two input representations are then prepared:

- **Flattened vectors:** each image becomes a vector of 784 features for Logistic Regression and the ANN.
- **Image tensors:** each image becomes a tensor with shape `(28, 28, 1)` for the CNN.

### 4. Trains a Logistic Regression baseline

The baseline uses scikit-learn's `LogisticRegression` with:

- `lbfgs` solver
- `max_iter=300`
- `random_state=42`
- Flattened, normalized image vectors as input

This model measures how well a relatively simple linear classifier performs before nonlinear or convolutional feature learning is introduced.

### 5. Trains an Artificial Neural Network

The ANN is a fully connected Keras model with this architecture:

```text
Input: 784 features
Dense: 128 units, ReLU
Dense: 64 units, ReLU
Output: 10 units, softmax
```

It is compiled with the Adam optimizer using a learning rate of `0.001`, sparse categorical cross-entropy loss, and accuracy as the tracked metric. Training runs for 10 epochs with a batch size of 64 while validation accuracy and loss are recorded.

### 6. Trains a Convolutional Neural Network

The CNN receives the original 2D image structure and uses this architecture:

```text
Input: 28 x 28 x 1
Conv2D: 32 filters, 3 x 3 kernel, ReLU, same padding
MaxPooling2D: 2 x 2
Conv2D: 64 filters, 3 x 3 kernel, ReLU, same padding
MaxPooling2D: 2 x 2
Flatten
Dense: 128 units, ReLU
Dense: 64 units, ReLU
Output: 10 units, softmax
```

The CNN uses the same Adam learning rate, loss function, batch size, and number of epochs as the ANN to make the comparison more consistent.

## Evaluation

Each model is evaluated on the unseen test set. The notebook reports:

- Test accuracy and loss.
- Weighted precision, recall, and F1-score.
- Per-class precision, recall, and F1-score.
- A confusion matrix showing which classes are confused with one another.
- Training and validation accuracy/loss curves for the ANN and CNN.
- Per-class F1-score comparisons across all three models.
- A side-by-side confusion-matrix dashboard.

The best model is selected programmatically using the highest test accuracy.

## Results

The benchmark values recorded in the notebook's final conclusions are:

| Model | Test accuracy | Role in the comparison |
| --- | ---: | --- |
| Logistic Regression | **84.27%** | Linear baseline using flattened pixels |
| ANN | **86.07%** | Nonlinear fully connected model |
| CNN | **89.01%** | Spatially aware image model |

The CNN performs best because convolution and pooling preserve and learn local spatial patterns such as edges, textures, and shape boundaries. The ANN improves on the linear baseline by learning nonlinear feature interactions, while Logistic Regression remains useful as a lightweight reference point.

The error analysis shows that visually similar categories, especially **Shirt**, **T-shirt/top**, **Pullover**, and **Coat**, are the most difficult to separate. More distinctive categories such as **Trouser** and **Bag** are generally classified more reliably.

## Requirements

The notebook requires Python 3 and the following packages:

```bash
pip install numpy pandas matplotlib scikit-learn keras tensorflow jupyter
```

Depending on the Keras installation, TensorFlow may be selected as the backend automatically. A GPU is optional; the models can also run on a CPU, although training will take longer.

## How to Run

1. Clone or download this repository.
2. Open `Fashion_MNIST_Logistic_ANN_CNN.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
3. Select a Python environment with the required packages installed.
4. Run the cells from top to bottom.
5. Wait for the dataset download and model-training cells to finish.
6. Review the printed metrics, plots, confusion matrices, and final comparison table.

The Fashion-MNIST dataset is downloaded through Keras when the dataset-loading cell runs, so no manual dataset download is required.

## Project Structure

```text
Fashion_MNIST_Logistic_ANN_CNN/
├── Fashion_MNIST_Logistic_ANN_CNN.ipynb  # Analysis, training, and evaluation
└── README.md                             # Project documentation
```

## Notes on Reproducibility

The notebook uses `keras.utils.set_random_seed(42)`, `random_state=42` for the Logistic Regression model and train/validation split, and fixed training settings. Exact neural-network results can still vary slightly between hardware, TensorFlow versions, and backend configurations.

## Possible Extensions

This comparison can be extended by adding data augmentation, dropout, batch normalization, early stopping, learning-rate scheduling, hyperparameter tuning, or additional models such as Random Forest, Support Vector Machines, or transfer-learning architectures.
