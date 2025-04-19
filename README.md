# Using Variational Autoencoder (VAE) for Outlier Detection

This project implements anomaly detection using Variational Autoencoders (VAE) and β-VAE. The goal is to identify outliers in a dataset through unsupervised learning techniques.

## Project Structure

- **main.py**: The main entry point for the application. It loads the dataset, processes it, and calls functions to perform anomaly detection using various methods, including VAE.
  
- **exp.py**: Contains experimental functions or classes that support the main functionality, including model training and evaluation.

- **beta_vae.py**: Implements the β-VAE model for anomaly detection. It defines the architecture of the β-VAE, including the encoder and decoder, and provides methods for training and inference.

- **data/U.csv**: The dataset file used for training and testing the anomaly detection models.

- **models/vae_model.py**: Contains the implementation of the standard VAE model, including its architecture and training procedures.

- **models/beta_vae_model.py**: Contains the specific implementation details for the β-VAE model, extending the functionality of the standard VAE.

- **utils/data_utils.py**: Includes utility functions for data preprocessing, loading, and manipulation.

- **utils/visualization.py**: Provides functions for visualizing the results of the anomaly detection, such as plotting graphs and figures.

- **results/figures**: Directory to store the output figures generated during the analysis and visualization of results.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd Using-Variational-Autoencoder-VAE-for-Outlier-Detection
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Ensure that the dataset `U.csv` is located in the `data` directory.

## Usage

To run the anomaly detection, execute the following command:
```
python main.py
```

This will load the dataset, preprocess it, and perform anomaly detection using both standard VAE and β-VAE methods. The results will be visualized and saved in the `results/figures` directory.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.