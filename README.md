# Lahore House Price Prediction System 🏠

An end-to-end Machine Learning solution for scraping, cleaning, training, evaluating, and predicting house prices in Lahore, Pakistan. Built using modern data engineering pipeline steps, multi-model evaluation benchmarks, and interactive user interfaces (including a Tkinter Desktop GUI and a Streamlit Web Dashboard).

---

## 🚀 Key Features

* **Data Engineering & Scraping:**
  * **Dynamic Web Crawler (`scraper.py`):** Uses Selenium WebDriver with smart scrolling behavior to scrape properties directly from Zameen.com.
  * **Intelligent Preprocessing (`preprocess.py`):** Standardizes pricing (PKR Crore/Lakh to integers) and area dimensions (Kanals, Sq Ft, Sq Yd to Marlas).
  * **Advanced Outlier Filtering:** Applies IQR-based filtering on price-per-marla and eliminates noisy records (e.g. houses with 50 bathrooms or years in 3020).
  * **Feature Engineering:** Creates 5 new predictive features including `property_age`, `total_rooms`, `luxury_score` (parking, servant quarters, drawing rooms), and `location_avg_price`.
  * **Automatic Location Aggregation:** Normalizes locations (e.g. spelling corrections) and groups rare locations (count < 10) into a single `"Other"` class to boost model generalization.

* **Machine Learning & Modeling:**
  * **Price Log-Transformation:** Fits target prices using `np.log1p` to handle highly skewed housing price distributions, and reverses predictions using `np.expm1` for inference.
  * **Baseline Comparison:** Integrates a Dummy Regressor baseline to verify model learning.
  * **5-Fold Cross-Validation:** Logs average $R^2$ and standard deviations.
  * **Advanced Evaluation Plots:** Saves residual charts, error breakdowns by price ranges, and feature importances.

* **User Interfaces:**
  * **Streamlit Web Application (`streamlit_app.py`):** Premium web-based interface with unit conversion calculators, similarity matchers, and interactive visualization charts.
  * **Tkinter Desktop GUI (`predict_gui.py`):** Responsive desktop widget containing inline error validations, combobox dropdown selections, reset options, and a results card.
  * **CLI Console Predictor (`predict_console.py`):** Command-line tool displaying PKR estimates and 3 similar properties in the console.

---

## 📊 Model Evaluation Results

After running the pipeline, the following benchmarks were generated from training on the cleaned dataset:

| Model Name | Test $R^2$ Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Mean Absolute % Error (MAPE) | 5-Fold Cross-Val $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline** | -0.0152 | Rs. 47,816,562 | Rs. 92,654,128 | 134.20% | -0.0210 $\pm$ 0.012 |
| **Linear Regression** | 0.2200 | Rs. 53,940,848 | Rs. 91,954,487 | 108.40% | 0.2104 $\pm$ 0.089 |
| **Decision Tree** | 0.5433 | Rs. 29,525,625 | Rs. 70,358,739 | 43.12% | 0.5218 $\pm$ 0.076 |
| **Random Forest** | 0.7185 | Rs. 24,413,436 | Rs. 55,239,471 | 38.65% | 0.7011 $\pm$ 0.045 |
| **Gradient Boosting** | 0.5763 | Rs. 29,118,094 | Rs. 67,769,897 | 45.33% | 0.5620 $\pm$ 0.061 |
| **XGBoost** | 0.6477 | Rs. 26,543,439 | Rs. 61,795,661 | 40.12% | 0.6310 $\pm$ 0.052 |
| **CatBoost (Best)** | **0.8062** | **Rs. 22,545,036** | **Rs. 45,839,275** | **31.25%** | **0.7980 $\pm$ 0.022** |

*Note: The model is off by approximately Rs. 2.2 Crore on average across a dataset featuring high-end locations like DHA Defence (where prices range from 4 to 40 Crore).*

---

## 🛠️ Setup & Installation

1. **Clone/Unpack** the repository files.
2. **Install package dependencies** using pip:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the main menu control panel:**
   ```bash
   python main.py
   ```

---

## 📖 Main Control Menu Options

When launching `main.py`, you can trigger any phase of the pipeline:
1. **Run scraper:** Re-scrape Zameen.com using headless Selenium.
2. **Run preprocessing:** Clean the raw CSV, filter outliers, and engineer features.
3. **Train models:** Benchmarks the regression models and serializes the best one.
4. **Predict house price (Console):** Console interface displaying price ranges and similar properties.
5. **Predict house price (GUI):** Desktop window application with validations.
6. **Run Streamlit Web Application:** Modern web application in the browser.
7. **Show model charts:** Saves baseline comparisons in `results/`.
8. **Evaluate best model:** Regenerates residual, distribution, and location pricing graphs.
9. **Exit:** Close program.
