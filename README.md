# Rice Leaf Disease Detection System

This project is a web-based application designed to detect diseases in rice leaves using deep learning. It leverages a pre-trained TensorFlow/Keras model to classify uploaded images and provides detailed descriptions along with precautionary measures for the identified diseases.

## Features

- **Image Upload Interface**: User-friendly web interface to upload rice leaf images.
- **Disease Detection**: Predicts the disease type from the uploaded image.
- **Confidence Score**: Displays the confidence level of the prediction.
- **Detailed Information**: Provides a description of the disease and lists necessary precautions.
- **Responsive Web Design**: Simple and effective UI using HTML/CSS.

## Project Structure

```
DP project/
├── dpproject/
│   ├── Rice_leaf_disease_detection.h5  # Pre-trained Keras model
│   ├── class_indices.json              # JSON mapping of class indices to names
│   └── description.csv                 # CSV containing disease descriptions and precautions
├── templates/
│   └── index.html                      # HTML template for the web interface
├── app.py                              # Main Flask application file
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

## Prerequisites

Ensure you have Python installed on your system. You will also need the following Python libraries, which are listed in `requirements.txt`:
- Flask
- pandas
- numpy
- tensorflow
- pillow

## Installation

1.  **Clone the valid repository or download the source code.**

2.  **Navigate to the project directory:**

    ```bash
    cd "path/to/DP project"
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *Note: It is recommended to use a virtual environment.*

4.  **Verify Data Files:**
    Ensure the following files are present in the `dpproject` directory:
    - `Rice_leaf_disease_detection.h5`
    - `class_indices.json`
    - `description.csv`

## Usage

1.  **Run the application:**

    ```bash
    python app.py
    ```

2.  **Access the web interface:**
    Open your web browser and go to:
    `http://127.0.0.1:5000/`

3.  **Detect Disease:**
    - Click on the "Choose File" button to upload an image of a rice leaf.
    - Click "Upload" or "Predict" (depending on your UI button label).
    - The application will display the predicted disease, confidence score, description, and precautions.

## Troubleshooting

-   **Model Not Found**: If you encounter a `FileNotFoundError`, ensure that the `dpproject` folder contains the model, JSON, and CSV files as expected by `app.py`.
-   **TensorFlow Compatibility**: Ensure your TensorFlow version is compatible with the Keras model.

## License

[MIT License](LICENSE) (or specify your license here)
