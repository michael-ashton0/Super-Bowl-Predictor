import requests
from bs4 import BeautifulSoup

#########################
#       SCRAPING        #
#########################

def get_header(stat, team):
    '''
    Preliminary function, run to get the header for your csv file
    Pass in the stat (ex 'passing') and a team and it will read the
    table's column names for the csv
    '''

    url = url = 'https://www.statmuse.com/nfl/ask/'

    response = requests.get(url + f'{team}-{stat}-in-week-1-2024') # Nobody has a week one bye, so I use week 1
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Weirdly named table makes it easy to find
        div_container = soup.find("div", class_="@container/table relative overflow-x-auto -mx-3")

        if div_container:
            thead = div_container.find('thead') # Table head?

            if thead:
                head_th_elements = thead.find_all('th')
                header = [th.get_text() for th in head_th_elements]
                header = ",".join(header)
            else:
                print("No thead found in the specified div.")
        else:
            print("No matching div found.")

    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return None
    
    return header[3:].replace(' ', '')

"""# Scraper(s) #"""

def get_stats(stat, team, week=1, year=2024):
    '''
    Query the statmuse website for the given stat, team, week, and year (ex 'passing','broncos', 1, 2024)
    Returns a csv string with the values of the table
    '''

    url = 'https://www.statmuse.com/nfl/ask/'

    response = requests.get(url + f'{team}-{stat}-in-week-{week}-{year}')

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        div_container = soup.find("div", class_="@container/table relative overflow-x-auto -mx-3")

        if div_container:
            # Table body
            tbody = div_container.find("tbody")

            if tbody:
                # Thankfully every value is in a td
                body_td_elements = tbody.find_all("td")
                data = [td.get_text(strip=True) for td in body_td_elements]

                # Format as CSV string
                csv_string = ",".join(data)

            else:
                print("No tbody found in the specified div.")
        else:
            print("No matching div found.")

    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return None
    
    return csv_string[3:]

def secondary_scraper(stat, team, week=1, year=2024):
    '''
    Sister function to get_stats, used exactly the same way
    but phrases its query slightly differently, for some reason
    the website sometimes understands one and not the other
    '''

    url = 'https://www.statmuse.com/nfl/ask/'

    response = requests.get(url + f'{team}-{stat}-week-{week}-{year}')

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        div_container = soup.find("div", class_="@container/table relative overflow-x-auto -mx-3")

        if div_container:

            thead = div_container.find('thead')
            tbody = div_container.find("tbody")

            if thead:
                head_th_elements = thead.find_all('th')
                header = [th.get_text() for th in head_th_elements]
                header = ",".join(header)

            if tbody:
                body_td_elements = tbody.find_all("td")
                data = [td.get_text(strip=True) for td in body_td_elements]

                csv_string = ",".join(data)
            else:
                print("No tbody found in the specified div.")
        else:
            print("No matching div found.")

    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return None
    
    return csv_string[3:]

#########################
#    GETTING STATS      #
#########################

def retrieve_data(team):
    '''
    Given a team, creates three files, one for their passing, rushing, and misc (first downs, 3rd%, and penalties)
    named statmuse_{team}_{stat}.csv
    '''

    for stat in ['passing', 'rushing']:
        with open(f'statmuse_{team}_{stat}.csv', 'w') as f:
            f.write(str(get_header(stat, team)) + '\n')
            for i in range(1, 18):
                data = get_stats(stat, team, i, 2024)
                if data is not None: # in case of bye week or bad request
                    f.write(str(data) + '\n') # newline for csv
                else:
                    print(f"Warning: No data found for {team} in week {i}")
        #print(f'{team} {stat} Done')

    # Misc has to be handled separately, since the query is made for penalties but file named misc
    with open(f'statmuse_{team}_misc.csv', 'w') as f:
        f.write(str(get_header('penalties', team)) + '\n')
        for i in range(1, 18):
            data = secondary_scraper('penalties', team, i, 2024)
            if data is not None:
                f.write(str(data) + '\n')
            else:
                print(f"Warning: No data found for {team} in week {i}")
    #print(f'{team} misc Done')

#########################
#    DATA PROCESSING    #
#########################

import pandas as pd

def prune_features(team):
    passing_df = pd.read_csv(f'statmuse_{team}_passing.csv')
    passing_df = passing_df.drop(columns=['Unnamed: 4', 'DATE', '.1', 'TD%', 'INT%', 'RTG', 'OPP', 'ATT'], errors='ignore')
    passing_df.rename(columns={'YDS ' : 'passYDS'}, inplace='TRUE')

    rushing_df = pd.read_csv(f'statmuse_{team}_rushing.csv')
    rushing_df = rushing_df.drop(columns=['Unnamed: 4', 'RESULT', 'TEAM', 'TM', 'DATE', '.1', 'TD%', 'INT%', 'RTG', 'OPP', 'ATT'], errors='ignore')
    rushing_df.rename(columns={'RUSH YDS ' : 'rushYDS'}, inplace='TRUE')

    misc_df = pd.read_csv(f'statmuse_{team}_misc.csv')
    misc_df.drop(columns=['DATE', 'SEASON', 'TM', 'Unnamed: 5', 'OPP', 'TD', 'OFF', 'OPPOFF', '4DWN%', 'RESULT','TEAM'], inplace=True)

    combined = pd.concat([passing_df, rushing_df, misc_df], axis=1)

    combined[['team_score', 'opponent_score']] = combined['RESULT'].str.extract(r'(\d+)-(\d+)')

    combined[['team_score', 'opponent_score']] = combined[['team_score', 'opponent_score']].astype(int)
    combined = combined.drop(columns=['TM', 'SCKY', 'opponent_score', 'RESULT'])
    combined.to_csv(f'{team}_stats.csv')

    return combined

# Create the combined stats file #

retrieve_data('chiefs')
retrieve_data('eagles')
print('All data successfully retrieved')

chiefs = prune_features('chiefs')
print('Chiefs pruned')
eagles = prune_features('eagles')
print('Eagles pruned')

prediction_data_2024 = pd.concat([chiefs,eagles])
prediction_data_2024.to_csv('prediction_data_2024.csv')

#########################
# END OF DATA ACQUIRING #
# MACHINE LEARNING BELOW#
#########################

import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('prediction_data_2024.csv')

# Normalize

scaler = MinMaxScaler()
df[['YDS', 'CMP', 'PCT', 'AVG', 'TD',
    'INT', 'SCK', 'PEN', '1DWN', '3DWN%',
    'RUSHYDS', 'YPC', 'RUSHTD']] = scaler.fit_transform(df[['YDS', 'CMP', 'PCT', 'AVG', 'TD',
    'INT', 'SCK', 'PEN', '1DWN', '3DWN%',
    'RUSHYDS', 'YPC', 'RUSHTD']])

# One Hot Encoding

encoder = OneHotEncoder(sparse_output=False, dtype=np.uint8)
encoded = encoder.fit_transform(df[['TEAM']])
columns = encoder.get_feature_names_out(['TEAM'])

encoded = pd.DataFrame(encoded, columns=columns)

df = pd.concat([df, encoded], axis=1)
df.drop(columns='TEAM', inplace=True)

# Setting up the data

X = df[['TEAM_Chiefs', 'TEAM_Eagles', 'YDS', 'CMP', 'PCT', 'AVG', 'TD',
    'INT', 'SCK', 'PEN', '1DWN', '3DWN%',
    'RUSHYDS', 'YPC', 'RUSHTD']]
y = df[['team_score']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Train the model

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).astype(np.uint8)

# Check the error
mae_score = mean_absolute_error(y_test, y_pred)
print('MAE: ', mae_score)

# Get their averages across the season per game
chiefs = pd.read_csv('chiefs_stats.csv')
eagles = pd.read_csv('eagles_stats.csv')

# Not sure where this comes from
chiefs = chiefs.drop(columns=['Unnamed: 0'])
eagles = eagles.drop(columns=['Unnamed: 0'])

for column in chiefs.columns:
    if column != 'TEAM':
        chiefs[column] = chiefs[column].mean()

for column in eagles.columns:
    if column != 'TEAM':
        eagles[column] = eagles[column].mean()

eagles.drop_duplicates(inplace=True)
chiefs.drop_duplicates(inplace=True)
final_data = pd.concat([chiefs, eagles])

final_data['TEAM_Eagles'] = [0,1]
final_data['TEAM_Chiefs'] = [1,0]

final_data[['YDS', 'CMP', 'PCT', 'AVG', 'TD',
    'INT', 'SCK', 'PEN', '1DWN', '3DWN%',
    'RUSHYDS', 'YPC', 'RUSHTD']] = scaler.transform(final_data[['YDS', 'CMP', 'PCT', 'AVG', 'TD',
    'INT', 'SCK', 'PEN', '1DWN', '3DWN%',
    'RUSHYDS', 'YPC', 'RUSHTD']])

superbowl = final_data[['TEAM_Chiefs', 'TEAM_Eagles', 'YDS', 'CMP', 'PCT', 'AVG', 'TD',
    'INT', 'SCK', 'PEN', '1DWN', '3DWN%',
    'RUSHYDS', 'YPC', 'RUSHTD']]

outcome = model.predict(superbowl)
outcome = outcome.flatten()

print(f'Chiefs : {int(outcome[0])} \nEagles : {int(outcome[1])}')