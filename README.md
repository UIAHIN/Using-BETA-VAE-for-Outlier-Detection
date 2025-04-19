# Using Variational Autoencoder (VAE) for Outlier Detection

This project implements anomaly detection using Variational Autoencoders (VAE) and β-VAE. The goal is to identify outliers in a dataset through unsupervised learning techniques.

## Project Structure

- **main.py**: The main entry point for the application. It loads the dataset, processes it, and calls functions to perform anomaly detection using various methods, including VAE.
  
- **exp.py**: Contains experimental functions or classes that support the main functionality, including model training and evaluation.

- **beta_vae.py**: Implements the β-VAE model for anomaly detection. It defines the architecture of the β-VAE, including the encoder and decoder, and provides methods for training and inference.


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

## Usage

To run the anomaly detection, execute the following command:
```
python main.py
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
