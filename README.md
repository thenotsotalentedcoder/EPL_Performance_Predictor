# Premier League Teams Performance Predictor

This Streamlit application predicts the performance of Premier League teams for the upcoming season based on historical data and machine learning models. It also visualizes historical team performances, including league positions and goals scored/conceded.

## Features

- Predicts team performance for the upcoming season, including:
  - Total points
  - Goals scored
  - Goals conceded
  - Predicted league rank
- Visualizes predicted performance metrics
- Displays historical performance charts:
  - Team rankings over past seasons
  - Goals scored and conceded over past seasons
- User-friendly interface for selecting teams and viewing predictions

## Technologies Used

- Python 3.x
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Altair (for data visualization)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/epl-predictor.git
   cd epl-predictor
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Ensure you have the necessary data files:
   - `preprocessed_data.pkl`: Preprocessed historical match data
   - `trained_models.pkl`: Trained machine learning models
   - `premier_league_standings.csv`: Historical Premier League standings data
   - Team logos in SVG format in the `logos/` directory

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`)

4. Select a team from the dropdown menu and click "Predict Performance" to see the predictions and visualizations for that team.

## How It Works

1. The app loads preprocessed historical match data and pre-trained machine learning models.
2. When a user selects a team, the app uses ensemble models (Random Forest, XGBoost, and LightGBM) to predict the team's performance for the upcoming season.
3. The app also predicts the performances of all other teams to estimate the selected team's league rank.
4. Historical data is used to create visualizations of past performance, allowing users to compare predictions with historical trends.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
