# %% [markdown]
# # Creating a Chatbot to ask questions about Movies
# You will use Netflix TV Shows and Movies data to create an ETL process to Extract, Transform and Load multiple datasets from CVS Files into a MongoDB Database. After that, you’ll use that data source to create a simple chatbot which allows the user to as a variety of questions to the chatbot. You DO NOT need to make this bot run in Discord or Twitter, but rather at the local machine.
#  
# Your bot will need to answer the questions (taking note to use various forms of ways to ask the question)in a human type of form.
# - What were the top 5 shows on Netflix 2 years ago? Show me the top 5 shows on Netflix 2 years ago. Show me the top 5 shows on Netflix two years ago.
# - What was the top movie on Netflix in 2020?
# - How long was the best movie on Netflix last year? What was the release year of that movie?
# 
# These are just *sample* questions. You need to allow you bot to ask 10 different categories/types of questions. They are up to you on which questions, but the bot needs to tell the user what those question categories that it can answer. Like: Top movies by year, top X movies / shows by year. Genre of Movie/Show of the top Movie/Show...# of seasons of top shows...etc. Star(s) of the top show/movie. You’ll need to use the user response to form a query for your Mongo Dataset.

# %% [markdown]
# ## My plan
# 1. Create Questions that people can ask the chatbox:
#     - *Question 1*: Ask for the highest rated movie for this year
#     - *Question 2*: Ask for the most popular movie for this year
#     - *Question 3*: Ask for the highest rated show for this year
#     - *Question 4*: Ask for the most popular show for this year
#     - *Question 5*: Ask for where the movie was produced (provide movie title)
#     - *Question 6*: Ask for movie genre (provide movie title)
#     - *Question 7*: Ask for movie runtime (provide movie title)
#     - *Question 8*: Ask for show runtime (provide show title)
#     - *Question 9*: Ask for what characters the actor has played (provide actor name)
#     - *Question 10*: Ask for movie age certification/rating (provide movie title)
# 2. Extract unique data for the answer to each question
# 3. Load each dataframe into a mongodb with each collection named after the question number
# 4. Create intents file and follow tensorflow <i>tag, patterns, responses, context_set</i> format
# 5. Train model in tensorflow to identify questions and responses
# 6. Create chat function to find prediction from tensorflow and query data based on tag and provide acceptable response 

# %% [markdown]
# ### Data Extraction and Transformation Layer
# 1. Extracting data from netflix (sourced from kaggle) [dataset](https://www.kaggle.com/datasets/thedevastator/the-ultimate-netflix-tv-shows-and-movies-dataset)
# 2. Transformed data to create unique tables for each of the above 10 questions with data that only pertains to answering the question

# %%
import pandas as pd
#load csv from url into pandas dataframe
best_movies_netflix = pd.read_csv('Best Movies Netflix.csv')
best_movies_years = pd.read_csv('Best Movie by Year Netflix.csv')
best_show_year = pd.read_csv('Best Show by Year Netflix.csv')
best_shows_netflix = pd.read_csv('Best Shows Netflix.csv')
raw_credits = pd.read_csv('raw_credits.csv')
raw_titles = pd.read_csv('raw_titles.csv')
#Question 1:
#Movie rating and release_year
movie_rating1 = best_movies_netflix[['TITLE', 'RELEASE_YEAR', 'SCORE']]
movie_rating2 = best_movies_years[['TITLE', 'RELEASE_YEAR', 'SCORE']]
movie_rating = (
    pd.concat([movie_rating1, movie_rating2])
    .drop_duplicates()
    .dropna()
    .reset_index(drop=True)
    .sort_values(by=['RELEASE_YEAR'])
    .reset_index(drop=True)
    .rename(columns={'TITLE': 'title', 'RELEASE_YEAR': 'release_year', 'SCORE': 'score'})
    .query('release_year == 2022')
    .nlargest(1, 'score')
)

# Question 2:
# Movie title, release_year, and NUMBER_OF_VOTES
popular_movies = (
    best_movies_netflix[['TITLE', 'RELEASE_YEAR', 'NUMBER_OF_VOTES']]
    .sort_values(by=['NUMBER_OF_VOTES'], ascending=False)
    .reset_index(drop=True)
    .rename(columns={'TITLE': 'title', 'RELEASE_YEAR': 'release_year', 'NUMBER_OF_VOTES': 'number_of_votes'})
    .query('release_year == 2022')
    .nlargest(1, 'number_of_votes')
)

#Question 3:
#Show title, release_year, rating
show_rating1 = best_shows_netflix[['TITLE', 'RELEASE_YEAR', 'SCORE']]
show_rating2 = best_show_year[['TITLE', 'RELEASE_YEAR', 'SCORE']]
show_rating = (
    pd.concat([show_rating1, show_rating2])
    .drop_duplicates()
    .dropna()
    .reset_index(drop=True)
    .sort_values(by=['RELEASE_YEAR'])
    .reset_index(drop=True)
    .rename(columns={'TITLE': 'title', 'RELEASE_YEAR': 'release_year', 'SCORE': 'score'})
    .query('release_year == 2022')
    .nlargest(1, 'score')
)

#Question 4:
#Show title, release_year, and NUMBER_OF_VOTES
popular_shows = (
    best_shows_netflix[['TITLE', 'RELEASE_YEAR', 'NUMBER_OF_VOTES']]
    .sort_values(by=['NUMBER_OF_VOTES'], ascending=False)
    .reset_index(drop=True)
    .rename(columns={'TITLE': 'title', 'RELEASE_YEAR': 'release_year', 'NUMBER_OF_VOTES': 'number_of_votes'})
    .query('release_year == 2022')
    .nlargest(1, 'number_of_votes')
)

#Question 5:
#runtime and movie title 
production_country = ( 
    raw_titles[['title', 'production_countries']]
    .dropna()
    .reset_index(drop=True)
    .rename(columns={'title': 'title', 'production_countries': 'production_countries'})
    # make production countries a string not list
    .assign(production_countries=lambda x: x.production_countries.astype(str))
    # remove brackets from production countries
    .assign(production_countries=lambda x: x.production_countries.str.replace('[', ''))
    .assign(production_countries=lambda x: x.production_countries.str.replace(']', ''))
    # remove quotes from production countries
    .assign(production_countries=lambda x: x.production_countries.str.replace("'", ''))
    # remove spaces from production countries
    .assign(production_countries=lambda x: x.production_countries.str.replace(" ", ' and '))
    # remove double ands from production countries
    .assign(production_countries=lambda x: x.production_countries.str.replace("and and", 'and'))
    # remove double commas from production countries
    .assign(production_countries=lambda x: x.production_countries.str.replace(",,", ','))
    # reset index
    .reset_index(drop=True)    
)
production_country
#Question 6:
#movie title and genre
movie_genre = (
    raw_titles[['title', 'genres']]
    .dropna()
    .reset_index(drop=True)
    .rename(columns={'title': 'title', 'genres': 'genres'})
)
movie_genre
#Question 7:
#movie title and runtime
#filter only movies from raw_titles
movie_titles = raw_titles[raw_titles['type'] == 'MOVIE']
movie_titles = movie_titles[['title', 'runtime']]
movie_titles
#Question 8:
#show title and runtime
#filter only shows from raw_titles
show_titles = raw_titles[raw_titles['type'] == 'SHOW']
show_titles = show_titles[['title', 'runtime']]
show_titles
#Question 9:
#actor name and characters played
actor_character =  (
    raw_credits[['name', 'character']]
    .dropna()
    .reset_index(drop=True)
    .rename(columns={'name': 'name', 'character': 'character'})
    .groupby('name')
    .agg({'character': ', '.join})
    .reset_index()
)
actor_character
#Question 10:
#age_certification and movie title
movie_certification = raw_titles[['title', 'age_certification']]
movie_certification = movie_certification.dropna()
movie_certification = movie_certification.reset_index(drop=True)
movie_certification = movie_certification.rename(columns={'title': 'title', 'age_certification': 'age_certification'})
print('dataframes created')

# %% [markdown]
# ### Data Loading Layer
# - using dataframes from above transforming them into tables in [MongoDB](https://www.mongodb.com/home)
# - database hosted locally

# %%
#import mongodb
from pymongo import MongoClient
#connect to mongodb
client = MongoClient('localhost', 27017)
#create database
db = client['movie_chatbot']
#create collection
movie_rating_collection = db['Question 1']
popular_movies_collection = db['Question 2']
show_rating_collection = db['Question 3']
popular_shows_collection = db['Question 4']
movie_runtime_collection = db['Question 5']
movie_genre_collection = db['Question 6']
movie_titles_collection = db['Question 7']
show_titles_collection = db['Question 8']
actor_character_collection = db['Question 9']
movie_certification_collection = db['Question 10']
#insert data into collection
movie_rating_collection.insert_many(movie_rating.to_dict('records'))
popular_movies_collection.insert_many(popular_movies.to_dict('records'))
show_rating_collection.insert_many(show_rating.to_dict('records'))
popular_shows_collection.insert_many(popular_shows.to_dict('records'))
movie_runtime_collection.insert_many(production_country.to_dict('records'))
movie_genre_collection.insert_many(movie_genre.to_dict('records'))
movie_titles_collection.insert_many(movie_titles.to_dict('records'))
show_titles_collection.insert_many(show_titles.to_dict('records'))
actor_character_collection.insert_many(actor_character.to_dict('records'))
movie_certification_collection.insert_many(movie_certification.to_dict('records'))
#check if data is inserted
print(
    movie_rating_collection.count_documents({}), 
    popular_movies_collection.count_documents({}), 
    show_rating_collection.count_documents({}), 
    popular_shows_collection.count_documents({}), 
    movie_runtime_collection.count_documents({}), 
    movie_genre_collection.count_documents({}), 
    movie_titles_collection.count_documents({}), 
    show_titles_collection.count_documents({}), 
    actor_character_collection.count_documents({}), 
    movie_certification_collection.count_documents({})
    )

#print all documents in collection
'''
for x in movie_rating_collection.find():
    print(x)
for x in popular_movies_collection.find():
    print(x)
for x in show_rating_collection.find():
    print(x)
for x in popular_shows_collection.find():
    print(x)
for x in movie_runtime_collection.find():
    print(x)
for x in movie_genre_collection.find():
    print(x)
for x in movie_titles_collection.find():
    print(x)
for x in show_titles_collection.find():
    print(x)
for x in actor_character_collection.find():
    print(x)
for x in movie_certification_collection.find():
    print(x)
'''
#close connection
client.close()
#check if connection is closed
print(client)
#check if database is closed
print(db)
print(f'Added to mongoDB')

# %% [markdown]
# ### Training Chatbot to Reply to given Questions 
# - creating JSON file (python dict) with given format
# ```JSON
# {"intents": [
#         {"tag": "greeting",
#          "patterns": ["Hi", "How are you", "Is anyone there?", "Hello", "Hey","Good day", "Whats up","Hola"],
#          "responses": ["Hello!", "Good to see you again!", "Hi there, how can I help?","hurry up, I don't have all day"],
#          "context_set": ""
#         },
#         {"tag": "goodbye",
#          "patterns": ["cya", "See you later", "Goodbye", "I am Leaving", "Have a Good day","bye"],
#          "responses": ["Sad to see you go..", "Talk to you later", "Goodbye!"],
#          "context_set": ""
#         }
#          
#    ]
# }
# ```
# - Using TensorFlow to train chatbot
# 

# %%
#create dictionary in python using format
#open mongodb connection
client = MongoClient('localhost', 27017)
#open database
db = client['movie_chatbot']
chatbot_train = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "How are you", "Is anyone there?", "Hello", "Hey","Good day", "Whats up","Hola"],
            "responses": ["Hello!", "Good to see you again!", "Hi there, how can I help?","hurry up, I don't have all day"],
            "context_set": ""
        },
        {
            "tag": "goodbye",
            "patterns": ["cya", "See you later", "Goodbye", "I am Leaving", "Have a Good day","bye"],
            "responses": ["Sad to see you go..", "Talk to you later", "Goodbye!"],
            "context_set": ""
        },
        {
            "tag": "Question 1",
            "patterns": [
                "What is the best movie of the year", 
                "What rating was the top movie from this year", 
                "What was the highest rated movie this year", 
                "What movie do critics like this year"
                ],
            "responses": [
                f"The movie rating is {db['Question 1'].find_one()['title']} with a rating of {db['Question 1'].find_one()['score']}", 
                f"The movie was given a rating of {db['Question 1'].find_one()['title']} with a rating of {db['Question 1'].find_one()['score']}", 
                f"The movie was rated {db['Question 1'].find_one()['title']} with a rating of {db['Question 1'].find_one()['score']} ",
                f"Critics gave this movie a rating of {db['Question 1'].find_one()['title']} with a rating of {db['Question 1'].find_one()['score']}"
                ],
            "context_set": ""
        },
        {
            "tag": "Question 2",
            "patterns": [
                "What is the most popular movie this year", 
                "What is the most popular movies right now", 
                "What is the most popular movies of today", 
                "What is the most popular movie"
                ],
            "responses": [
                f"The most popular movie is {db['Question 2'].find_one()['title']} with a rating of {db['Question 2'].find_one()['number_of_votes']}",
                f"The most popular movie right now is {db['Question 2'].find_one()['title']} with a rating of {db['Question 2'].find_one()['number_of_votes']}", 
                f"The most popular movie of today is {db['Question 2'].find_one()['title']} with a rating of {db['Question 2'].find_one()['number_of_votes']}",
                f"The most popular movie is {db['Question 2'].find_one()['title']} with a rating of {db['Question 2'].find_one()['number_of_votes']}"
                ],
            "context_set": ""
        },
        {
            "tag": "Question 3",
            "patterns": [
                "What is the rating of the", 
                "What rating was the show given", 
                "How was this show rated", 
                "Do critics like this show"
                ],
            "responses": [
                f"The show rating is {db['Question 3'].find_one()['title']} with a rating of {db['Question 3'].find_one()['score']}",
                f"The show was given a rating of {db['Question 3'].find_one()['title']} with a rating of {db['Question 3'].find_one()['score']}", 
                f"The show was rated {db['Question 3'].find_one()['title']} with a rating of {db['Question 3'].find_one()['score']}", 
                f"Critics gave this show a rating of {db['Question 3'].find_one()['title']} with a rating of {db['Question 3'].find_one()['score']}"
                ],
            "context_set": ""
        },
        {
            "tag": "Question 4",
            "patterns": [
                "What are the most popular shows", 
                "What are the most popular shows right now", 
                "What are the most popular shows of all time", 
                "What are the most popular shows of the year"
                ],
            "responses": [
                f"The most popular shows are {db['Question 4'].find_one()['title']} with a rating of {db['Question 4'].find_one()['number_of_votes']}",
                f"The most popular shows right now are {db['Question 4'].find_one()['title']} with a rating of {db['Question 4'].find_one()['number_of_votes']}", 
                f"The most popular shows of all time are {db['Question 4'].find_one()['title']} with a rating of {db['Question 4'].find_one()['number_of_votes']}",
                f"The most popular shows of the year are {db['Question 4'].find_one()['title']} with a rating of {db['Question 4'].find_one()['number_of_votes']}"
                ],
            "context_set": ""
        },
        {
            "tag": "Question 5",
            "patterns": [
                "Where was produced", 
                "production country", 
                "country of origin", 
                "where was the movie produced"
                ],
            "responses": [
                "The movie was produced in ", 
                "This movie was shot in ", 
                "The movie was produced in ",
                "The movie production is from ",
                ],
            "context_set": ""
        },
        {
            "tag": "Question 6",
            "patterns": [
                "What category", 
                "What genre ", 
                "What is the style ", 
                "What genre does the movie fall under"
                ],
            "responses": [
                "The movie genre is ", 
                "The movie category is ", 
                "The genre of this movie is ", 
                "The movie follows the genre of "
                ],
            "context_set": ""
        },
        {
            "tag": "Question 7",
            "patterns": [
                "How long ", 
                "What is the runtime ", 
                "How many minutes movie", 
                "How long is movie"
                ],
            "responses": [
                "The movie runtime is ", 
                "The movie runs for ", 
                "The movie is ", 
                "The runtime of the movie is "
                ],
            "context_set": ""
        },
        {
            "tag": "Question 8",
            "patterns": [
                "How long is the show", 
                "How many minutes is the show ", 
                "What is the runtime for the show ", 
                "show runs for how long "
                ],
            "responses": [
                "The show runtime is ", 
                "The show runs for ", 
                "The show is ", 
                "The show of the movie is "
                ],
            "context_set": ""
        },
        {
            "tag": "Question 9",
            "patterns": [
                "What did the actor play",
                "Who was he acting as", 
                "Who was she acting as", 
                "Who has played",
                ],
            "responses": [
                "They played the character ", 
                "They played the role of ", 
                "They acted as the character ", 
                "They performed as the character "
                ],
            "context_set": ""
        },
        {
            "tag": "Question 10",
            "patterns": [
                "What is the movie age rating", 
                "What is the movie certification", 
                "What is the movie restriction", 
                "Is this movie appropriate for children"
                ],
            "responses": [
                "The movie age rating is ", 
                "The movie certification is ", 
                "The movie restriction is ", 
                "This movie has an age rating of "
                ],
            "context_set": ""
        }
    ]
}
#download this dictionary as intents.json
import json
with open('intents.json', 'w') as outfile:
    json.dump(chatbot_train, outfile)
print("JSON file created")

# %%
import nltk 
nltk.download('punkt')

from nltk import word_tokenize,sent_tokenize

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
#read more on the steamer https://towardsdatascience.com/stemming-lemmatization-what-ba782b7c0bd8
import numpy as np 
import tflearn
import tensorflow as tf
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle","rb") as f:
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
training_shape = np.shape(training)
print(f"Input data shape: {training_shape}")

# Check the dimensions of the output data
output_shape = np.shape(output)
print(f"Output data shape: {output_shape}")



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
print("model created")

# %%
def chat(inp):

    
    if inp.lower() == "quit":
        print("Thank you for using the Movie Chatbot! Goodbye!")
        return False
    if inp.lower() == "help":
        print(" Question 1: Ask for the highest rated movie for this year\n",
        "Question 2: Ask for the most popular movie for this year\n",
        "Question 3: Ask for the highest rated show for this year\n",
        "Question 4: Ask for the most popular show for this year\n",
        "Question 5: Ask for where the movie was produced (provide movie title)\n",
        "Question 6: Ask for movie genre (provide movie title)\n",
        "Question 7: Ask for movie runtime (provide movie title)\n",
        "Question 8: Ask for show runtime (provide show title)\n",
        "Question 9: Ask for what characters the actor has played (provide actor name)\n",
        "Question 10: Ask for movie age certification/rating (provide movie title)\n"
        )
        return True
    result = model.predict([bag_of_words(inp, words)])[0]
    result_index = np.argmax(result)
    tag = labels[result_index]

    if result[result_index] > 0.7:
        for tg in data["intents"]:
            if tg['tag'] == tag and tag == "Question 1":
                responses = tg['responses']
                print(f'Bot: {random.choice(responses)} \n')
                return True
            elif tg['tag'] == tag and tag == "Question 2":
                responses = tg['responses']
                print(f'Bot: {random.choice(responses)} \n')
                return True
            elif tg['tag'] == tag and tag == "Question 3":
                responses = tg['responses']
                print(f'Bot: {random.choice(responses)} \n')
                return True
            elif tg['tag'] == tag and tag == "Question 4":
                responses = tg['responses']
                print(f' Bot: {random.choice(responses)} \n')
                return True
            elif tg['tag'] == tag and tag == "Question 5":
                responses = tg['responses']
                for title in db['Question 5'].find():
                    if title['title'] in inp:
                        name_of_movie = title['title']
                        production_countries = ' and '.join(db["Question 5"].find_one({"title": name_of_movie})["production_countries"])
                        print(f'Bot: {random.choice(responses)} {production_countries} \n')
                        return True
                print(f"Bot: I didnt get that. Can you explain or try again.\n")
                return True
            elif tg['tag'] == tag and tag == "Question 6":
                responses = tg['responses']
                for title in db['Question 6'].find():
                    if title['title'] in inp:
                        name_of_movie = title['title']
                        genres = ' and '.join(db["Question 6"].find_one({"title": name_of_movie})["genres"])
                        print(f'Bot: {random.choice(responses)} {genres}\n')
                        return True
                print(f"Bot: I didnt get that. Can you explain or try again.\n")
                return True
            elif tg['tag'] == tag and tag == "Question 7":
                responses = tg['responses']
                for title in db['Question 7'].find():
                    if title['title'] in inp:
                        name_of_movie = title['title']
                        minutes = db["Question 7"].find_one({"title": name_of_movie})["runtime"]
                        print(f'Bot: {random.choice(responses)} {minutes} minutes.\n')
                        return True
                print(f"Bot: I didnt get that. Can you explain or try again.\n")
                return True
            elif tg['tag'] == tag and tag == "Question 8":
                responses = tg['responses']
                for title in db['Question 8'].find():
                    if title['title'] in inp:
                        name_of_movie = title['title']
                        runtime = db["Question 8"].find_one({"title": name_of_movie})["runtime"]
                        print(f'{random.choice(responses)} {runtime} minutes.\n')
                        return True
                print(f"Bot: I didnt get that. Can you explain or try again.\n")
                return True
            elif tg['tag'] == tag and tag == "Question 9":
                responses = tg['responses']
                for actor in db['Question 9'].find():
                    if actor['name'] in inp:
                        actor_name = actor['name']
                        characters = ' and '.join(db["Question 9"].find_one({"name": actor_name})["characters"])
                        print(f'Bot: {random.choice(responses)} {characters}\n')
                        return True
                print(f"Bot: I didnt get that. Can you explain or try again.\n")
                return True
            elif tg['tag'] == tag and tag == "Question 10":
                responses = tg['responses']
                for title in db['Question 10'].find():
                    if title['title'] in inp:
                        name_of_movie = title['title']
                        age_certification = db["Question 10"].find_one({"title": name_of_movie})["age_certification"]
                        print(f'Bot: {random.choice(responses)} {age_certification}\n')
                        return True
                print(f"Bot: I didnt get that. Can you explain or try again.\n")
                return True
    else:
        print(f"Bot: I didnt get that. Can you explain or try again.\n")
        return True

# %%
print('='*50)
print("Welcome to the Movie Chatbot!")
print('='*50)
print("Type help to see the list of acceptable question topics")
print("Type quit to exit the program")
while True:
    inp = input("You: ")
    if not chat(inp):
        break
    


