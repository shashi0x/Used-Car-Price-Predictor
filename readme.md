# Used Car Price Prediction

## Overview
This project focuses on predicting the prices of used cars by building an end-to-end **data pipeline** and **machine learning model** trained on real-world listings scraped from [CarWale](https://www.carwale.com/).  
The workflow includes **web scraping**, **data preprocessing**, **feature engineering**, and **deep learning** model training.

---

## Project Structure
OLDCAR  
│  
├── data  
│ ├── CarData.csv # Raw scraped data  
│ ├── carwale.html # Sample HTML page  
│ ├── CleanDataNP.csv # Processed numeric dataset  
│ ├── dataReadme.md # Data documentation  
│ ├── jsonEx.json # Extracted JSON sample  
│ └── scrapping.ipynb # Web scraping + data collection  
│  
├── model  
│ |── CleanDataNP.csv # Final preprocessed dataset  
│ ├── linear.ipynb # Linear regression attempt  
│ └── neural.ipynb # Neural network model  
├── readme.md


## Dataset
- **Source:** Scraped from CarWale’s used car listings across **350+ Indian cities**.  
- **Size:** 8,718 examples.  
- **Collection Method:**  
  - Extracted car listing details from JSON objects embedded in `<script>` tags.  
  - For diversity, 30 listings were scraped from each city’s page instead of paginating deeper.  

### Dataset Statistics
- **Training set:** 6,000 examples  
- **Test set:** 2,718 examples  
- **Mean Price:** ₹19,15,637  
- **Median Price:** ₹8,00,000  
- **Min Price:** ₹25,000  
- **Max Price:** ₹4,21,00,000  
- **Std Dev:** ₹31,13,977  

---

## Data Preprocessing
- Extracted raw data (HTML → JSON → CSV).  
- Handled categorical variables by encoding them into numeric values (e.g., body type, transmission type).  
- Applied **scaling** for numerical stability.  
- Converted dataset to **NumPy arrays** for training.  

---

## Modeling
### 1. Linear Regression (Baseline)
- Tried linear regression first.  
- Model failed to capture complex relationships in data.  

### 2. Neural Network (Final Model)
Implemented using **TensorFlow/Keras**:

```python
model = Sequential([
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(256, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])
```
## Results & Conclusion

* Seen Data MAE: ₹3,23,231

* Unseen Data MAE: ₹3,68,877

* While the MAE is still high, it is reasonable given the dataset includes both budget cars and ultra-luxury cars (₹25,000 to ₹4.2 Cr range).

* A single model struggles because predicting a Swift and a BMW i7 with the same function is naturally imbalanced.

* Key learning: Instead of training one global model, grouping cars (e.g., budget, mid-range, luxury) and training separate models would likely yield better accuracy.

#### This demonstrates an important ML lesson: sometimes data segmentation and domain knowledge can be more impactful than just tuning architectures.

## Tech Stack

* Language: Python

* Libraries: NumPy, Pandas, Requests, BeautifulSoup, JSON, TensorFlow/Keras

## Learnings

* Designing and implementing a web scraper for structured data collection.

* Parsing and extracting data from embedded JSON objects.

* Applying feature engineering (categorical encoding + scaling).

* Comparing baseline ML (Linear Regression) vs Deep Learning.

* Understanding dataset imbalance and how it affects ML model performance.

## Future Work

* Split dataset into groups (budget / premium / luxury) and train specialized models.

* Improve preprocessing with outlier detection and feature importance analysis.

* Experiment with alternative models (XGBoost, Random Forest, or even hybrid ensembles).

* Build a simple deployment interface (API or Streamlit app).

## Author

### Shashi Kumar
**Student | Exploring AI/ML | Interested in practical applications of Machine Learning**
