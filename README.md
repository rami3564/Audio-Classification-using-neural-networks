# Male/Female Audio-Classification-using-neural-networks

**Task:**
> To train a custom model to classify songs into 4 categories (male, female, both, none)

**Overview**

This project focuses on the classification of audio data using a custom ResNet inspired model tailored to work with Mel spectrograms. The repository contains the code and resources necessary to preprocess audio files, train a ResNet-based neural network, and evaluate its performance on the given dataset. This project was part of a Kaggle competition at my college where we had to achieve 80% accuracy on the test dataset. With this model, I was able to achieve >99% accuracy on my training dataset and ~80% accuracy on the test dataset.

**Key Features**

_Data Preprocessing:_ Utilizes librosa for loading audio files and Spleeter for vocal isolation. Implements a comprehensive audio processing pipeline to standardize and enhance audio data.

_Data Augmentation:_ Includes techniques such as time-shifting to augment the training data and improve model robustness.

_Model Architecture:_ Custom ResNet inspired model designed to classify Mel spectrograms with an accuracy of up to 99% on training dataset.

_Training and Evaluation:_ Scripts to train the model using GPU acceleration and evaluate its performance.

**Project Structure**
_data:_ Directory to store audio files.
 
_notebooks:_ Jupyter notebooks for exploratory data analysis and experiments.
 
_src:_ Source code for data preprocessing, model definition, training, and evaluation.

_models:_ Directory to save trained models.

_results:_ Directory to save evaluation results and model predictions.

**Requirements**
Python 3.7+
Libraries: librosa, Spleeter, torch, numpy, scikit-learn, matplotlib
GPU for training (recommended: Google Colab with GPU support)

**Installation**
1. Clone the repository
2. Install the required packages

**Results**
> The model achieved a maximum accuracy of >99% on training dataset
> Detailed logs and visualizations of training and evaluation are available in the results directory

**Challenges and Learnings**
> Encountered significant computational challenges due to hardware limitations
> Utilized Google Colab's GPU services but faced data upload constraints and computation quotas


**Contributing**
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

