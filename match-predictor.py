# Import necessary modules and libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, render_template

# Load the data and information about the teams and players
data = pd.read_csv('teams_and_players.csv')

# Define the features and target for the model
features = ['goals_scored', 'goals_conceded', 'shots_on_target', 'pass_accuracy']
target = 'winner'

# Train the predictive model using logistic regression
model = LogisticRegression()
model.fit(data[features], data[target])

# Function to predict the winner of a game
def predict_winner(team1, team2):
    # Retrieve the stats for the teams
    stats1 = data[data['team_name'] == team1][features]
    stats2 = data[data['team_name'] == team2][features]
    
    # Use the model to predict the winner
    if model.predict(stats1) == 1:
        return team1
    elif model.predict(stats2) == 1:
        return team2
    else:
        return 'Draw'

# Create a Flask app
app = Flask(__name__)

# Route to render the home page
@app.route('/')
def home():
    return render_template('home.html')

# Route to predict the winner of a game
@app.route('/predict_winner', methods=['POST'])
def predict_winner_route():
    team1 = request.form['team1']
    team2 = request.form['team2']
    winner = predict_winner(team1, team2)
    return render_template('result.html', team1=team1, team2=team2, winner=winner)

# Run the app
if __name__ == '__main__':
    app.run()
