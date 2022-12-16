##### This program is to create the mongoDB
# Get packages ready
import pandas as pd
import pymongo
import pprint

# Extract and Transform 
titles = pd.read_csv(r"raw_titles.csv").drop(["index"], axis = 1)
credit = pd.read_csv(r"raw_credits.csv")

all_data = titles.merge(credit, left_on = "id", right_on = "id", how = "left")

# Load into a mongoDB
host_name = "localhost"
port = "27017"

conn_str = {"local" : f"mongodb://{host_name}:{port}/"}
client = pymongo.MongoClient(conn_str["local"])

db_name = "Data_Project_2"
db = client[db_name]
netflix = db.netflix
netflix.insert_many(all_data.to_dict("records"))


