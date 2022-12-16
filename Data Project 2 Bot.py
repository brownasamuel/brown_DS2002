##### This program is to run the chatbot
import nltk 
nltk.download('punkt')
from nltk import word_tokenize, sent_tokenize
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
#read more on the steamer https://towardsdatascience.com/stemming-lemmatization-what-ba782b7c0bd8
import numpy as np 
import tflearn
import tensorflow as tf
import random
import json
import pickle
import re
#########
with open(r"Project bot intents.json") as file:
    data = json.load(file, strict = False)

try:
    with open(r"data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
            
        if intent["tag"] not in labels:
            labels.append(intent["tag"])


    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
               bag.append(1)
            else:
              bag.append(0)
    
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        
        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)
    
    with open("data.pickle","wb") as f:
        pickle.dump((words, labels, training, output), f)



net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    
    return np.array(bag)
#########
def chat():
    print("Hello! You can ask me any of these questions, but please enter numbers and years using numerals(E.g. 5 instead of five): \n 1: What was the top rated movie in ___(year)? \n 2: What were the top 5 countries for movie production in ____(year)? \n 3: How many movies and shows has ____(actor) been in that are on Netflix? \n 4: What is the longest movie of ____(genre) in the database? \n 5: What is the earliest movie with a rating of __(age rating)? \n 6: How many seasons did the top-rated show from ____(year) go? \n 7: What __(number) shows got the most votes on IMDB? \n 8: What was the most common genre for a show between ____ and ____(years)? \n 9: What is the IMDB score of the best voted show with a rating of ____(age rating)? \n 10: Who are the __(number) highest-rated movie directors? \n Type 'help' for assistance \n Type 'quit' to stop")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        
        result = model.predict([bag_of_words(inp, words)])[0]
        result_index = np.argmax(result)
        tag = labels[result_index]

        if result[result_index] > 0.7:
            if tag == "greeting":
                responses = data["intents"][0]["responses"]
                print(random.choice(responses))
            elif tag == "goodbye":
                responses = data["intents"][1]["responses"]
                print(random.choice(responses))
            elif tag == "help":
                responses = data["intents"][2]["responses"]
                print(random.choice(responses))
                
            elif tag == "Question 1":
                year = int(re.findall(r'\d+', inp)[0])
                query = {"$and" : [{"type" : "MOVIE"}, {"release_year" : year}]}
                results = list(netflix.find(query).sort("imdb_score", -1).limit(1))
                if len(results) == 0:
                    print("This year doesn't have a movie in the dataset. Please try again")
                else:
                    print(f"The best movie of {year} per IMDB scores is: {results[0]['title']}, with a score of {results[0]['imdb_score']}.")
                
            elif tag == "Question 2":
                year = int(re.findall(r'.*([1-3][0-9]{3})', inp)[0])
                query = {"$and" : [{"type" : "MOVIE"}, {"release_year" : year}]}, {"title" : 1, "production_countries" : 1}
                results = list(netflix.find({"$and" : [{"type" : "MOVIE"}, {"release_year" : year}]}, {"title" : 1, "production_countries" : 1, "_id" : 0}))
                if len(results) > 0:
                    results = pd.DataFrame(results).drop_duplicates()
                    results["production_countries"] = results["production_countries"].apply(eval)
                    results_list = results.explode("production_countries")["production_countries"].value_counts().index.tolist()[0:5]
                    print(f"The top countries in {year} were: {', '.join([str(x + 1) + '. ' + results_list[x] for x in range(len(results_list))])}.")
                else:
                    print("This year did not have enough movies to answer your question. Please try again")
                
            elif tag == "Question 3":
                inp2 = re.sub(r'[^\w\s-]', ' ', (inp + " ")).title()
                inp3 = re.sub("(Movies |Movie |Shows |Show |What |Has |How |Many |Been |In |Often |Acted |When |And |On |Netflix |That |Are |A |Was |Is |But |Are |Or |I |IMDB |Starred )", "", inp2)
                name_words = [x for x in inp3.split() if len(list(netflix.find(
                    {"$and": [{"name" : {"$regex" : x}}, {"role" : "ACTOR"}]}))) > 0]
                actor = " ".join(name_words)
                roles = list(netflix.find({"$and": [{"name" : {"$regex" : actor}}, {"role" : "ACTOR"}]}))
                if len(actor.split()) == 1:
                    if len(roles) == 0:
                        print("This actor could not be found in the database.")
                    elif len(pd.DataFrame(roles).name.unique()) == 1:
                        movie_roles = [x for x in roles if x['type'] == "MOVIE"]
                        show_roles = [x for x in roles if x['type'] == "SHOW"]
                        movie_ct = len(movie_roles) 
                        show_ct = len(show_roles)
                        print(f"{actor} has been in {movie_ct} movies and {show_ct} shows on Netflix.")
                    elif len(pd.DataFrame(roles).name.unique()) > 1:
                        print("Please reenter the actor's full name and check spelling! There is more than one actor with the name you entered.")
                elif len(actor.split()) > 1:
                    movie_ct = len([x for x in roles if x['type'] == "MOVIE"])
                    show_ct = len([x for x in roles if x['type'] == "SHOW"])
                    if movie_ct == 0 and show_ct == 0:
                        print("This actor/actress has no shows or movies in the database.")
                    elif len(actor.split()) == 1:
                        print("Please enter the actor's full name")
                    else:
                        print(f"{actor} has been in {movie_ct} movies and {show_ct} shows on Netflix.")
                else:
                    print("Please try again, and make sure the spelling of your actor's name is correct.")
                
            elif tag == "Question 4":
                inp2 = re.sub(r'[^\w\s]', ' ', inp).lower()
                if "documentary" in inp2:
                    inp2 = "documentation"
                genre = [x for x in inp2.split() if len(list(netflix.find({"$and" : [{"type" : "MOVIE"}, {"genres" : {"$regex" : "'" + x + "'"}}]}))) > 0]
                
                if len(genre) == 1:
                    if genre[0] == "documentary":
                        genre[0] = "documentation"
                    answer = list(netflix.find({"$and" : [{"type" : "MOVIE"}, {"genres" : {"$regex" : genre[0]}}]}).sort("runtime", -1).limit(-1))[0]
                    print(f"The longest {genre[0]} movie was a {answer['release_year']} film called {answer['title']}.")
                elif len(genre) != 1:
                    print("Please reenter your question with one genre spelled correctly.")
            elif tag == "Question 5":
                inp2 = re.sub(r'[^\w\s-]', ' ', inp ).upper()
                rating = [x for x in inp2.split() if len(list(netflix.find({"$and" : [{"type" : "MOVIE"}, {"age_certification" : x}]}))) > 0]
                if len(rating) == 1:
                    answer = list(netflix.find({"$and" : [{"type" : "MOVIE"}, {"age_certification" : rating[0]}]}).sort("release_year").limit(1))[0]
                    print(f"The first {answer['age_certification']} movie in the dataset was released in {answer['release_year']}, and was called {answer['title']}.")
                else:
                    print("Please reenter your question, and make sure that the age rating you are entering is spelled correctly.")
            elif tag == "Question 6":
                year = re.findall(r'\d+', inp)
                if len(year) == 1:
                    year = int(year[0])
                    query = {"$and" : [{"type" : "SHOW"}, {'release_year' : year}]}
                    results = list(netflix.find(query).sort("imdb_score", -1).limit(1))
                    if len(results) == 1:
                        print(f"The top-rated show of {year} was {results[0]['title']}, which ran for {results[0]['seasons']} seasons.")
                    else:
                        print("This year doesn't appear to have any shows.")
                else:
                    print("Please reenter your question with a year in numerals.")
                
            elif tag == "Question 7":
                num_shows_str = re.findall(r'\d+', inp)
                if len(num_shows_str) == 0:
                    print("Please reenter your question and include the digit of the number of shows that you would like to see this time.")
                else:
                    num_shows = int(num_shows_str[0])
                    top_shows = pd.DataFrame(list(netflix.find({"type" : "SHOW"}, {"title" : 1, "imdb_votes" : 1, "_id" : 0}).sort("imdb_votes", -1))).drop_duplicates(ignore_index = True)[0:num_shows]
                    top_shows_list = top_shows['title'].tolist()
                    print(f"The top {num_shows} shows by imdb votes were(in order from most to least): {', '.join([str(x + 1) + '. ' + top_shows_list[x] for x in range(len(top_shows_list))])}")
            elif tag == "Question 8":
                years = re.findall(r'\d+', inp)
                if len(years) < 2:
                    print("Please reenter your question with a correct period of time from which to get the most common genre.")
                else:
                    years = [int(x) for x in years]
                    y1 = min(years)
                    y2 = max(years)
                    shows = list(netflix.find({"$and" : [{"type" : "SHOW"}, {"release_year" : {"$gte": y1}}, {"release_year" : {"$lte": y2}}]}, {"title" : 1, "genres" : 1, "_id" : 0}))
                    shows_df = pd.DataFrame(shows).drop_duplicates(ignore_index = True)
                    genres_list = [re.sub(r'[^\w\s]', '', str(x)) for x in shows_df["genres"]]
                    answer = pd.Series(" ".join(genres_list).split()).value_counts().index[0]
                    print(f"The most common genre in TV shows from {y1} to {y2} was {answer}.")
               
            elif tag == "Question 9":
                inp2 = re.sub(r'[^\w\s-]', ' ', inp ).upper()
                rating = [x for x in inp2.split() if len(list(netflix.find({"$and" : [{"type" : "SHOW"}, {"age_certification" : x}]}))) > 0]
                if len(rating) == 1:
                    answer = list(netflix.find({"$and" : [{"type" : "SHOW"}, {"age_certification" : rating[0]}]}).sort("imdb_score", -1).limit(1))[0]
                    print(f"The best voted show with an age rating of {rating[0]} was {answer['title']}, with a score of {answer['imdb_score']}.")
                else:
                    print("Please reenter your question, and make sure that the age rating you are entering is spelled correctly.")
            elif tag == "Question 10":
                num_dir_str = re.findall(r'\d+', inp)
                if len(num_dir_str) == 0:
                    print("Please reenter your question and include the digit of the number of directors that you would like to see this time.")
                else:
                    num_dir = int(num_dir_str[0])
                    top_dir = pd.DataFrame(list(netflix.find({"$and" : [{"type" : "MOVIE"}, {"role" : "DIRECTOR"}]}, {"name" : 1, "imdb_score" : 1, "_id" : 0}))).groupby("name").mean().sort_values("imdb_score", ascending = False)[0:num_dir]
                    top_dir_list = top_dir.index.tolist()
                    print(f"The top {num_dir} by average imdb scores were(in order from highest to lowest): {', '.join([str(x + 1) + '. ' + top_dir_list[x] for x in range(len(top_dir_list))])}")
        else:
            print("Sorry, I didn't get that! Please try again. You can ask me any of these questions: \n 1: What was the top rated movie in ___(year)? \n 2: What were the top 5 countries for movie production in ____(year)? \n 3: How many movies and shows has ____(actor) been in that are on Netflix? \n 4: What is the longest movie of ____(genre) in the database? \n 5: What is the earliest movie with a rating of __(age rating)? \n 6: How many seasons did the top-rated show from ____(year) go? \n 7: What __(number) shows got the most votes on IMDB? \n 8: What was the most common genre for a show between ____ and ____(years)? \n 9: What is the IMDB score of the best voted show with a rating of ____(age rating)? \n 10: Who are the __(number) highest-rated movie directors? \n Type 'help' for assistance \n Type 'quit' to stop")

            
chat()