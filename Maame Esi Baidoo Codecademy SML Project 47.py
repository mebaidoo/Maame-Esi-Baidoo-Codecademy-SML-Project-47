import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
tennis_stats = pd.read_csv("tennis_stats.csv")
print(tennis_stats.head())

# perform exploratory analysis here:
#Plotting different features against the different outcomes to find if there is any relationship between them
plt.scatter(tennis_stats["Aces"], tennis_stats["Wins"])
plt.title("Wins Vs. Aces")
plt.show()
plt.clf()
plt.scatter(tennis_stats["ServiceGamesWon"], tennis_stats["Wins"])
plt.title("Wins Vs. ServiceGamesWon")
plt.show()
plt.clf()
plt.scatter(tennis_stats["DoubleFaults"], tennis_stats["Wins"])
plt.title("Wins Vs. DoubleFaults")
plt.show()
plt.clf()
plt.scatter(tennis_stats["BreakPointsOpportunities"], tennis_stats["Wins"])
plt.title("Wins Vs. BreakPointsOpportunities")
plt.show()
plt.clf()
#There is a strong positive relationship between the BreakPointsOpportunities and Wins, also between DoubleFaults and Wins

## perform single feature linear regressions here:
#Using BreakPointsOpportunities to predict Wins
#Splitting the data
break_points = tennis_stats["BreakPointsOpportunities"]
break_points = break_points.values.reshape(-1, 1)
wins = tennis_stats["Wins"]
wins = wins.values.reshape(-1, 1)
break_train, break_test, win_train, win_test = train_test_split(break_points, wins, train_size = 0.8, test_size = 0.2)
#Creating the model
regr = LinearRegression()
regr.fit(break_train, win_train)
score = regr.score(break_test, win_test)
print("Score for break points: " + str(score))
win_predicted = regr.predict(break_test)
#print(win_predicted)
#Plotting the model’s predictions on the test set against the actual outcome variable to visualize the performance
plt.scatter(win_test, win_predicted, alpha = 0.4)
plt.title("Break Points and Wins Model")
plt.xlabel("Wins")
plt.ylabel("Wins Predicted")
plt.show()

#Performing the same linear regression model on DoubleFaults to predict Wins
double_faults = tennis_stats["DoubleFaults"]
double_faults = double_faults.values.reshape(-1, 1)
wins = tennis_stats["Wins"]
wins = wins.values.reshape(-1, 1)
double_train, double_test, win_train, win_test = train_test_split(double_faults, wins, train_size = 0.8, test_size = 0.2)
#Creating the model
regr_2 = LinearRegression()
regr_2.fit(double_train, win_train)
score_double = regr_2.score(double_test, win_test)
print("Score for double faults : " + str(score_double))
win_predicted_2 = regr_2.predict(double_test)
#print(win_predicted_2)
#Plotting the model’s predictions on the test set against the actual outcome variable to visualize the performance
plt.clf()
plt.scatter(win_test, win_predicted_2, alpha = 0.4)
plt.title("Double Faults and Wins Model")
plt.xlabel("Wins")
plt.ylabel("Wins Predicted")
plt.show()
#BreakPointsOpportunities was better at predicting Wins than DoubleFaults, probably because of the strong relationship

## perform two feature linear regressions here:
#Using BreakPointsOpportunities and FirstServeReturnPointsWon to predict Winnings
break_first = tennis_stats[["BreakPointsOpportunities", "FirstServeReturnPointsWon"]]
winnings = tennis_stats["Winnings"]
break_first_train, break_first_test, winnings_train, winnings_test = train_test_split(break_first, winnings, train_size = 0.8, test_size = 0.2)
#Creating the model
regr_3 = LinearRegression()
regr_3.fit(break_first_train, winnings_train)
score_break_first = regr_3.score(break_first_test, winnings_test)
print("Score for break points and first serve : " + str(score_break_first))
winnings_predicted = regr_3.predict(break_first_test)
#Plotting the model’s predictions on the test set against the actual outcome variable to visualize the performance
plt.clf()
plt.scatter(winnings_test, winnings_predicted, alpha = 0.4)
plt.title("Two factors (break point and first) and Winnings Model")
plt.xlabel("Winnings")
plt.ylabel("Winnings Predicted")
plt.show()

#Using ServiceGamesWon and ReturnGamesWon to predict Winnings
serv_return = tennis_stats[["ServiceGamesWon", "ReturnGamesWon"]]
winnings = tennis_stats["Winnings"]
serv_return_train, serv_return_test, winnings_train, winnings_test = train_test_split(serv_return, winnings, train_size = 0.8, test_size = 0.2)
#Creating the model
regr_4 = LinearRegression()
regr_4.fit(serv_return_train, winnings_train)
score_serv_return = regr_4.score(serv_return_test, winnings_test)
print("Score for service and return points won : " + str(score_serv_return))
winnings_predicted_2 = regr_4.predict(serv_return_test)
#Plotting the model’s predictions on the test set against the actual outcome variable to visualize the performance
plt.clf()
plt.scatter(winnings_test, winnings_predicted_2, alpha = 0.4)
plt.title("Two factors (service and return points) and Winnings Model")
plt.xlabel("Winnings")
plt.ylabel("Winnings Predicted")
plt.show()
#BreakPointsOpportunities and FirstServeReturnPointsWon was far better at predicting than ServiceGamesWon and ReturnGamesWon
## perform multiple feature linear regressions here:
#Using service game columns (Offensive) to predict yearly earnings
service = tennis_stats[["Aces", "DoubleFaults", "FirstServe", "FirstServePointsWon", "SecondServePointsWon", "BreakPointsFaced", "BreakPointsSaved", "ServiceGamesPlayed", "ServiceGamesWon", "TotalServicePointsWon"]]
winnings = tennis_stats["Winnings"]
service_train, service_test, winnings_train, winnings_test = train_test_split(service, winnings, train_size = 0.8, test_size = 0.2)
#Creating the model
regr_5 = LinearRegression()
regr_5.fit(service_train, winnings_train)
score_service = regr_5.score(service_test, winnings_test)
print("Score for service: " + str(score_service))
winnings_predicted_serv = regr_5.predict(service_test)
#Plotting the model’s predictions on the test set against the actual outcome variable to visualize the performance
plt.clf()
plt.scatter(winnings_test, winnings_predicted_serv, alpha = 0.4)
plt.title("Multiple factors (service) and Winnings Model")
plt.xlabel("Winnings")
plt.ylabel("Winnings Predicted")
plt.show()

#Using return game columns (Defensive) to predict yearly earnings
defensive = tennis_stats[["FirstServeReturnPointsWon", "SecondServeReturnPointsWon", "BreakPointsOpportunities", "BreakPointsConverted", "ReturnGamesPlayed", "ReturnGamesWon", "ReturnPointsWon", "TotalPointsWon"]]
winnings = tennis_stats["Winnings"]
defensive_train, defensive_test, winnings_train, winnings_test = train_test_split(defensive, winnings, train_size = 0.8, test_size = 0.2)
#Creating the model
regr_6 = LinearRegression()
regr_6.fit(defensive_train, winnings_train)
score_defensive = regr_6.score(defensive_test, winnings_test)
print("Score for defensive: " + str(score_defensive))
winnings_predicted_def = regr_6.predict(defensive_test)
#Plotting the model’s predictions on the test set against the actual outcome variable to visualize the performance
plt.clf()
plt.scatter(winnings_test, winnings_predicted_def, alpha = 0.4)
plt.title("Multiple factors (defensive) and Winnings Model")
plt.xlabel("Winnings")
plt.ylabel("Winnings Predicted")
plt.show()
#Service (Offensive) is better at predicting Winnings than Return (Defensive)