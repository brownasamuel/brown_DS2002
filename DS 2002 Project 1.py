##### Code also available on github at https://github.com/brownasamuel/brown_DS2002

# Import packages
import requests
import json
import pandas as pd
import re
import sqlite3
#%% Getting the data
# Choose your team! # Enter a team's MLB id, which is a number from 108-121 or 133-147(Ex: 121 to get the Mets). Then, run the rest of this cell and proceed. 
team_id = input() 
if int(team_id) not in list(range(108, 122)) and int(team_id) not in list(range(133, 148)):
    print("Please reenter your team id according to the directions")

urlquote = f"http://lookup-service-prod.mlb.com/json/named.roster_40.bam?team_id='{team_id}'"
header_var = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

response = requests.get(urlquote, headers = header_var)    
#%% Converting the data format
# Enter the format in which you would like the data(JSON, CSV, or SQL). 
output_format = input() 

if re.search("CSV", output_format.upper()):
    output = pd.DataFrame(response.json()["roster_40"]["queryResults"]["row"])
elif re.search("JSON", output_format.upper()):
    output = response.json()["roster_40"]["queryResults"]["row"]
elif re.search("SQL", output_format.upper()):
    # Make the sqlite DB
    conn = sqlite3.connect('team_roster')
    cur = conn.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS roster (college, end_date, pro_debut_date, status_code, name_full, name_display_first_last, birth_date, height_inches, team_id, name_last, bats, player_id, position_txt, primary_position, jersey_number, starter_sw, start_date, name_display_last_first, name_first, name_use, weight, throws, team_name, team_code, team_abbrev, height_feet)')
    conn.commit()
    # Get the info ready to put in the DB 
    output = pd.DataFrame(response.json()["roster_40"]["queryResults"]["row"])
    output.to_sql('roster', conn, if_exists = 'replace', index = False)
else:
    print("Please re-enter your preferred format, and make sure that you spelled it correctly")


#%% Modifying the columns
# Here, run the code to combine the two height columns and remove several more. 
if re.search("CSV", output_format.upper()):
    output["height"] = output["height_feet"].astype(str) + "'" + output["height_inches"].astype(str)
    output = output.drop(columns = ["end_date", "status_code", "team_code", "height_feet", "height_inches"], axis = 1)
elif re.search("JSON", output_format.upper()):
    remove = ["end_date", "status_code", "team_code", "height_feet", "height_inches"]
    for i in range(len(output)):
        output[i]["height"] = output[i]["height_feet"] + "'" + output[i]["height_inches"]
        [output[i].pop(key) for key in remove]
elif re.search("SQL", output_format.upper()):
    # Get data ready to re-insert
    output["height"] = output["height_feet"].astype(str) + "'" + output["height_inches"].astype(str)
    output = output.drop(columns = ["end_date", "status_code", "team_code", "height_feet", "height_inches"], axis = 1)
    # Replace old data with new
    cur.execute("DROP TABLE roster")
    conn.commit()
    output.to_sql('roster', conn, if_exists = 'replace', index = False)
    conn.close()
else:
    print("Please use the same output_format for all sections of this code, and do them in order")


#%% Writing to local file
# Here, run the code to receive your output file in your specified format. 
if re.search("CSV", output_format.upper()):
    output.to_csv("team_roster.csv")
elif re.search("JSON", output_format.upper()):
    with open("team_roster.json", "w") as file:
        json.dump(output, file)
elif re.search("SQL", output_format.upper()):
    print("The sqlite database should already be in your working directory")
else:
    print("Please use the same output_format for all sections of this code, and do them in order")


#%% Brief summary of the data file(including number of records and columns)
# Run this code to learn a little about your data. 
if re.search("CSV", output_format.upper()) or re.search("SQL", output_format.upper()):
    print(f"This file contains {len(output)} rows and {len(output.columns)} columns for a total of {output.size} elements. It contains information on each player on the 40-man roster of the {output.team_name[0]}, including their height, weight, birthdate, and primary position.")
elif re.search("JSON", output_format.upper()):
    print(f"This file contains {len(output)} rows and {len(output[0].keys())} columns for a total of {len(output[0].keys()) * len(output)} elements. It contains information on each player on the 40-man roster of the {output[0]['team_name']}, including their height, weight, birthdate, and primary position.")
else:
    print("Please use the same output_format for all sections of this code, and do them in order")

