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
import joblib
import pickle

warnings.filterwarnings('ignore')

def engineer_features(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['MonthOfYear'] = df['Date'].dt.month
    df['IsHomeGame'] = np.where(df['HomeTeam'] == df['Team'], 1, 0)
    df['GoalDifference'] = df['GF'] - df['GA']

    df['xG'] = df['GF'] * np.random.normal(1, 0.1, len(df))
    df['xGA'] = df['GA'] * np.random.normal(1, 0.1, len(df))
    df['xGDifference'] = df['xG'] - df['xGA']

    df['Points'] = np.select(
        [df['GF'] > df['GA'], df['GF'] == df['GA'], df['GF'] < df['GA']],
        [3, 1, 0],
        default=0
    )

    metrics = ['GF', 'GA', 'xG', 'xGA', 'Points']
    for metric in metrics:
        df[f'{metric}_ExpandingAvg'] = df.groupby('Team')[metric].transform(
            lambda x: x.expanding().mean()
        )

    return df

def train_ensemble_model(X, y):
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    models = [
        RandomForestRegressor(n_estimators=100, random_state=42),
        XGBRegressor(n_estimators=100, random_state=42),
        LGBMRegressor(n_estimators=100, random_state=42),
    ]

    trained_models = []
    for model in models:
        model.fit(X_train, y_train)
        trained_models.append(model)

    ensemble_predictions = np.mean([model.predict(X_test) for model in trained_models], axis=0)

    print(f"MAE: {mean_absolute_error(y_test, ensemble_predictions)}")
    print(f"RMSE: {mean_squared_error(y_test, ensemble_predictions, squared=False)}")

    return trained_models, imputer

def predict_team_performance(team_data, team_name, next_season, models_dict):
    last_season = team_data['Season'].max()

    if len(team_data[team_data['Season'] == last_season]) == 0:
        print(f"No data found for {team_name} in the last season. Cannot make predictions.")
        return None

    features = ['DayOfWeek', 'MonthOfYear', 'IsHomeGame', 'xG', 'xGA',
                'GF_ExpandingAvg', 'GA_ExpandingAvg', 'xG_ExpandingAvg',
                'xGA_ExpandingAvg', 'Points_ExpandingAvg']

    next_season_data = team_data[team_data['Season'] == last_season].copy()
    next_season_data['Season'] = next_season
    next_season_data['MonthOfYear'] = 8  # Assuming season starts in August

    X = models_dict['imputer'].transform(next_season_data[features])

    predicted_goals_scored = np.mean([model.predict(X) for model in models_dict['goals_scored']], axis=0).mean() * 38
    predicted_goals_against = np.mean([model.predict(X) for model in models_dict['goals_against']], axis=0).mean() * 38
    predicted_points = np.mean([model.predict(X) for model in models_dict['points']], axis=0).mean() * 38

    return {
        'Total Points': round(predicted_points),
        'Goals for': round(predicted_goals_scored),
        'Goals Against': round(predicted_goals_against),
    }

def predict_rank(df, all_teams, next_season, models_dict):
    rankings = []
    for team in all_teams:
        team_data = df[df['Team'] == team]
        prediction_result = predict_team_performance(team_data, team, next_season, models_dict)
        if prediction_result is not None:
            rankings.append((team, prediction_result['Total Points']))
        else:
            rankings.append((team, df['Points'].mean()))

    rankings.sort(key=lambda x: x[1], reverse=True)
    team_ranks = {team: min(rank, 20) for rank, (team, points) in enumerate(rankings, 1)}
    return team_ranks

def preprocess_and_save_data(input_csv_path, output_pickle_path):
    df = pd.read_csv(input_csv_path)

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

    with open(output_pickle_path, 'wb') as f:
        pickle.dump(df, f)

    print(f"Preprocessed data saved to {output_pickle_path}")

@st.cache_resource
def load_preprocessed_data(pickle_path):
    with open(pickle_path, 'rb') as f:
        df = pickle.load(f)
    return df

@st.cache_data
def load_historical_data():
    return pd.read_csv("premier_league_standings.csv")

def train_and_save_models(df, models_pickle_path):
    features = ['DayOfWeek', 'MonthOfYear', 'IsHomeGame', 'xG', 'xGA',
                'GF_ExpandingAvg', 'GA_ExpandingAvg', 'xG_ExpandingAvg',
                'xGA_ExpandingAvg', 'Points_ExpandingAvg']
    X = df[features]

    models_dict = {}
    models_dict['goals_scored'], models_dict['imputer'] = train_ensemble_model(X, df['GF'])
    models_dict['goals_against'], _ = train_ensemble_model(X, df['GA'])
    models_dict['points'], _ = train_ensemble_model(X, df['Points'])

    with open(models_pickle_path, 'wb') as f:
        pickle.dump(models_dict, f)

    print(f"Trained models saved to {models_pickle_path}")

@st.cache_resource
def load_trained_models(models_pickle_path):
    with open(models_pickle_path, 'rb') as f:
        models_dict = pickle.load(f)
    return models_dict

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

    data_pickle_path = "preprocessed_data.pkl"
    models_pickle_path = "trained_models.pkl"

    df = load_preprocessed_data(data_pickle_path)
    models_dict = load_trained_models(models_pickle_path)
    historical_data = load_historical_data()

    all_teams = sorted(df['Team'].unique())
    next_season = df['Season'].max() + 1

    selected_team = st.selectbox("Select a team:", all_teams)

    if st.button("Predict Performance"):
        team_data = df[df['Team'] == selected_team]
        prediction_result = predict_team_performance(team_data, selected_team, next_season, models_dict)
        
        if prediction_result is not None:
            team_ranks = predict_rank(df, all_teams, next_season, models_dict)

            col1, col2 = st.columns([1, 2])

            with col1:
                logo_path = f"logos/{selected_team.lower().replace(' ', '_')}.svg"
                if os.path.exists(logo_path):
                    st.image(logo_path, width=150)
                else:
                    st.write("Team logo not found")

            with col2:
                st.subheader(f"Predictions for {selected_team}")
                st.write("Season: 2024-2025")
                st.write(f"Predicted Points: {prediction_result['Total Points']}")
                st.write(f"Predicted Goals Scored: {prediction_result['Goals for']}")
                st.write(f"Predicted Goals Conceded: {prediction_result['Goals Against']}")
                st.write(f"Predicted Rank: {team_ranks[selected_team]}")

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
