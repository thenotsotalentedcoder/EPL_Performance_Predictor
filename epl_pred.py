import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import os
import altair as alt

warnings.filterwarnings('ignore')


# Feature Engineering Function (Updated for your data)
def engineer_features(df):
    df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date is datetime
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['MonthOfYear'] = df['Date'].dt.month
    df['IsHomeGame'] = np.where(df['HomeTeam'] == df['Team'], 1, 0)
    df['GoalDifference'] = df['GF'] - df['GA']

    # Basic xG/xGA estimation (since you don't have these columns)
    df['xG'] = df['GF'] * np.random.normal(1, 0.1, len(df))
    df['xGA'] = df['GA'] * np.random.normal(1, 0.1, len(df))
    df['xGDifference'] = df['xG'] - df['xGA']

    # Calculate points
    df['Points'] = np.select(
        [df['GF'] > df['GA'], df['GF'] == df['GA'], df['GF'] < df['GA']],
        [3, 1, 0],
        default=0
    )

    # Recent performance metrics (using expanding window)
    metrics = ['GF', 'GA', 'xG', 'xGA', 'Points']
    for metric in metrics:
        df[f'{metric}_ExpandingAvg'] = df.groupby('Team')[metric].transform(
            lambda x: x.expanding().mean()
        )

    return df

# Train Ensemble Model Function
def train_ensemble_model(X, y):
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # Ensemble of diverse models
    models = [
        RandomForestRegressor(n_estimators=100, random_state=42),
        XGBRegressor(n_estimators=100, random_state=42),
        LGBMRegressor(n_estimators=100, random_state=42),
    ]

    # Train each model and make predictions
    predictions = []
    for model in models:
        model.fit(X_train, y_train)
        predictions.append(model.predict(X_test))

    # Average predictions from all models
    ensemble_predictions = np.mean(predictions, axis=0)

    # Evaluate the ensemble model performance
    print(f"MAE: {mean_absolute_error(y_test, ensemble_predictions)}")
    print(f"RMSE: {mean_squared_error(y_test, ensemble_predictions, squared=False)}")

    return lambda X: np.mean([model.predict(X) for model in models], axis=0)  # Return prediction function

# Prediction Function (Updated)
def predict_team_performance(df, team_name, next_season):
    team_data = df[df['Team'] == team_name].copy()
    last_season = team_data['Season'].max()

    # Check if there is data for the selected team in the last season
    if len(team_data[team_data['Season'] == last_season]) == 0:
        print(f"No data found for {team_name} in the last season. Cannot make predictions.")
        return None  # Return None to indicate that predictions cannot be made

    features = ['DayOfWeek', 'MonthOfYear', 'IsHomeGame', 'xG', 'xGA',
                'GF_ExpandingAvg', 'GA_ExpandingAvg', 'xG_ExpandingAvg',
                'xGA_ExpandingAvg', 'Points_ExpandingAvg']

    X = team_data[features]

    # Train models for goals scored, goals against, and points
    goals_scored_model = train_ensemble_model(X, team_data['GF'])
    goals_against_model = train_ensemble_model(X, team_data['GA'])
    points_model = train_ensemble_model(X, team_data['Points'])

    # Predict for the next season (simplified - assumes similar schedule)
    next_season_data = team_data[team_data['Season'] == last_season].copy()
    next_season_data['Season'] = next_season
    next_season_data['MonthOfYear'] = 8  # Assuming season starts in August

    predicted_goals_scored = goals_scored_model(next_season_data[features]).mean() * 38  # Adjust for 38 games
    predicted_goals_against = goals_against_model(next_season_data[features]).mean() * 38  # Adjust for 38 games
    predicted_points = points_model(next_season_data[features]).mean() * 38  # Adjust for 38 games

    return {
        'Total Points': round(predicted_points),
        'Goals for': round(predicted_goals_scored),
        'Goals Against': round(predicted_goals_against),
    }

# Function to predict rank based on predicted points (Updated with rank limit)
def predict_rank(df, all_teams, next_season):
    rankings = []
    for team in all_teams:
        prediction_result = predict_team_performance(df, team, next_season)
        if prediction_result is not None:
            rankings.append((team, prediction_result['Total Points']))
        else:
            # Handle teams not in predictions (e.g., newly promoted) - assign average points
            rankings.append((team, df['Points'].mean()))

    rankings.sort(key=lambda x: x[1], reverse=True)
    team_ranks = {team: min(rank, 20) for rank, (team, points) in enumerate(rankings, 1)}  # Limit rank to 20
    return team_ranks


@st.cache_data
def load_and_process_data():
    df = pd.read_csv("deduplicated_full_match_data.csv")

    home_df = df.copy()
    home_df['Team'] = home_df['HomeTeam']
    home_df['GF'] = home_df['FTHG']
    home_df['GA'] = home_df['FTAG']

    away_df = df.copy()
    away_df['Team'] = away_df['AwayTeam']
    away_df['GF'] = away_df['FTAG']
    away_df['GA'] = away_df['FTHG']

    df = pd.concat([home_df, away_df], ignore_index=True)
    df['Season'] = df['Date'].str.split('-').str[0].astype(int)
    df = engineer_features(df)

    return df

@st.cache_data
def load_historical_data():
    return pd.read_csv("premier_league_standings.csv")

def create_ranking_chart(historical_data, team_name):
    team_data = historical_data[historical_data['team'] == team_name]
    chart = alt.Chart(team_data).mark_line(point=True).encode(
        x='season_end_year:O',
        y=alt.Y('position:Q', scale=alt.Scale(reverse=True), title='Position'),
        tooltip=['season_end_year', 'position']
    ).properties(
        title=f"{team_name}'s Premier League Rankings",
        width=600,
        height=400
    )
    return chart

def create_goals_chart(historical_data, team_name):
    team_data = historical_data[historical_data['team'] == team_name]
    goals_data = pd.melt(team_data, id_vars=['season_end_year'], value_vars=['gf', 'ga'], 
                         var_name='goal_type', value_name='goals')
    goals_data['goal_type'] = goals_data['goal_type'].map({'gf': 'Goals For', 'ga': 'Goals Against'})
    
    chart = alt.Chart(goals_data).mark_line(point=True).encode(
        x='season_end_year:O',
        y='goals:Q',
        color='goal_type:N',
        tooltip=['season_end_year', 'goal_type', 'goals']
    ).properties(
        title=f"{team_name}'s Goals Scored and Conceded",
        width=600,
        height=400
    )
    return chart

def main():
    st.set_page_config(page_title="EPL Predictor", layout="wide")

    st.title("Premier League Teams Performance Predictor")

    # Load and process data
    df = load_and_process_data()
    historical_data = load_historical_data()

    # Get list of teams and next season
    all_teams = sorted(df['Team'].unique())
    next_season = df['Season'].max() + 1

    # Create dropdown for team selection
    selected_team = st.selectbox("Select a team:", all_teams)

    if st.button("Predict Performance"):
        # Make prediction
        prediction_result = predict_team_performance(df, selected_team, next_season)
        
        if prediction_result is not None:
            team_ranks = predict_rank(df, all_teams, next_season)

            # Display results
            col1, col2 = st.columns([1, 2])

            with col1:
                # Display team logo
                logo_path = f"logos/{selected_team.lower().replace(' ', '_')}.svg"
                if os.path.exists(logo_path):
                    st.image(logo_path, width=150)
                else:
                    st.write("Team logo not found")

            with col2:
                st.subheader(f"Predictions for {selected_team}")
                st.write(f"Season: {next_season}-{next_season+1}")
                st.write(f"Predicted Points: {prediction_result['Total Points']}")
                st.write(f"Predicted Goals Scored: {prediction_result['Goals for']}")
                st.write(f"Predicted Goals Conceded: {prediction_result['Goals Against']}")
                st.write(f"Predicted Rank: {team_ranks[selected_team]}")

            # Visualize the prediction
            st.subheader("Performance Visualization")
            
            chart_data = pd.DataFrame({
                'Metric': ['Points', 'Goals Scored', 'Goals Conceded'],
                'Value': [prediction_result['Total Points'], prediction_result['Goals for'], prediction_result['Goals Against']]
            })

            chart = alt.Chart(chart_data).mark_bar().encode(
                x='Metric',
                y='Value',
                color='Metric'
            ).properties(width=500, height=300)

            st.altair_chart(chart, use_container_width=True)

            # Historical performance charts
            st.subheader("Historical Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                ranking_chart = create_ranking_chart(historical_data, selected_team)
                st.altair_chart(ranking_chart, use_container_width=True)
            
            with col2:
                goals_chart = create_goals_chart(historical_data, selected_team)
                st.altair_chart(goals_chart, use_container_width=True)

        else:
            st.error(f"Unable to make predictions for {selected_team}.")

if __name__ == "__main__":
    main()
