# Social Media Analysis

This is a Streamlit-based web application that performs sentiment analysis on Facebook comments. It allows users to enter a Facebook page ID and access token, and then displays the sentiment analysis results, model performance metrics, and a visualization of the sentiment distribution.

## Features

- Fetch comments from a Facebook page using the Graph API
- Perform sentiment analysis on the comments using a pre-trained logistic regression model
- Display the sentiment analysis results in a table
- Visualize the sentiment distribution of the analyzed comments

## Prerequisites

- Python 3.7 or higher
- Required Python packages:
  - `streamlit`
  - `requests`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `emoji`

## Installation

1. Clone the repository: `git clone https://github.com/your-username/social-media-analysis.git`

2. Navigate to the project directory: `cd social-media-analysis`

3. Install the required packages: `pip install -r requirements.txt`

## Usage

1. Run the Streamlit application: `streamlit run app.py`

2. The web application will open in your default web browser.

3. Enter the Facebook page ID and access token in the form, then click the "Analyze" button.

4. The application will display the sentiment analysis results, model performance metrics, and a visualization of the sentiment distribution.

5. You can download the sentiment analysis results as a CSV file using the "Download Results" button.

## Contributing

If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---

### Developer Information

- **Name**: Ankit Kumar Singh
- **Email**: [ansingh0174@gmail.com](ansingh0174@gmail.com)

### Dataset Credit

The sentiment analysis model was trained using a dataset sourced from Kaggle. Credits to the original dataset creators can be found [here](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset?select=train.csv).
