# DS 3002 Final Project: Discord Bot (August Lamb/abl6ywp)


import os
import discord
import requests
import random

import datetime as dt

from dotenv import load_dotenv


load_dotenv()  # Load environment variables

TOKEN = os.getenv('DISCORD_TOKEN')

GUILD = os.getenv('GUILD_ID')


class MyClient(discord.Client):

    '''Discord client, handles all bot logic'''
    async def on_ready(self):
        '''Brings bot online'''
        for guild in self.guilds:
            if guild.name == GUILD:
                break
    print(
        f'{self.user} has connected to Discord!\n',
        f'{guild.name} (ID: {guild.id})'
    )


    async def on_message(self, message):
        '''Handles reply logic when bot is tagged in a message'''


                 def checkforhelp(msg):

                        '''Checks to see if the word help is in the message'''


                         return "help" in msg


                 def checkevent(msg):

                        '''Checks param string to determine correct event format for swimcloud api'''


                        res = ""  # Accumulator for event format string


                        strokes = {  # Dict of possible names for any given event

    ("free", "freestyle", "freestyler", "fr", "mile", "miler"): "1",

                                ("back", "backstroke", "backstroker", "bk"): "2",

                                ("breast", "breaststroke", "breaststroker", "br"): "3",

                                ("fly", "butterfly", "butterflier", "butterflyer", "flyer", "flier"): "4",

                                ("im", "medley", "imer", "im'er", "medlier", "medleyer"): "5"

                        }



                        distances = { # Dict of possible names for any given event distance

    ("50", "fifty"): "50",

                                ("100", "one"): "100",

                                ("200", "two"): "200",

                                ("500", "five"): "500",

                                ("1000", "thousand"): "1000",

                                ("1650", "mile", "miler"): "1650"

                        }


                        possibilities = [  # All possible event format strings

    "150", "1100", "1200", "1500", "11000", "11650",    # Freestyle

    "250", "2100", "2200",                              # Backstroke

    "350", "3100", "3200",                              # Breaststroke

    "450", "4100", "4200",                              # Butterfly

    "5100", "5200"                                      # IM

    ]


                         for stk in strokes: # Checks to see if there is a match for a stroke name in the message

                                 for pos in stk:

                                         for wrd in msg:

                                                 if pos == wrd:

                                                        res += strokes[stk]

                                                         print(f"Identified stroke {res} ({wrd})")

                                                         break


                         for distance in distances: # Checks to see if there is a match for a stroke distance in the message

                                 for dist in distance:

                                         for wrd in msg:

                                                 if dist == wrd:

                                                        res += distances[distance]

                                                         print(f"Identified distance {distances[distance]}")

                                                         break


                         if res in possibilities: # Double checks to see if the combination in the accumulator is a possible event in the api

                                 print(f"Found event: {res}")

                                 return res

                         else:

                                 print("Could not find event, expect an error!")

                                 return None


                 def checkgender(msg):

                        '''Checks message for indicators of gender'''


                        female = ["woman", "women", "female", "girl", "girls", "womens"]  # list of possible names for gender female

                        male = ["man", "men", "mens", "male", "boys", "boy", "guy", "guys"]  # list of possible names for gender male


                         for wrd in msg: # Checks each word in message to see if theres a match in one of the lists above

                                 for gen in female:

                                         if wrd == gen: # If there's a match, return the proper format for swimcloud api

                                                 print("Identified gender female")

                                                 return "F"

                                 for gen in male:


                                         if wrd == gen:

                                                 print("Identified gender male")

                                                 return "M"

                    

                         print("Could not identify gender, expect an error!")

                         return None


                 def makeshift_ner(msg):

                        '''"Makeshift" named entity recognition. Drives the checkgender and checkevent functions.

            Checks message for entities required for the swimcloud api.'''


                        querystring = {  # Pre-formatted querystring for swimcloud api (missing gender and event)

    "gender": "",

    "event": "",

    "region": "countryorganisation_usacollege",

    "season_id": "25",

    "page": "1",

    "eventcourse": "Y"

    }


                        gen = checkgender(msg)  # Analyze message for gender and event entities

                        evt = checkevent(msg)


                         if gen == None or evt == None: # If either of the entities is not found, the task cannot be completed

                                 return None


                         else:

                                querystring["gender"] = gen  # If both are found, insert to the querystring and return

                                querystring["event"] = evt

                                 return querystring


                 def getSwimData(msg):

                        '''Analyzes message to format an HTTP request to swimcloud for the message's query.'''


                        url = 'http://www.swimcloud.com/api/splashes/top_times/'  # swimcloud api url

                        header = {}

                        querystring = makeshift_ner(msg)  # get querystring from makeshift_ner function


                         if querystring == None: # if makeshift_ner returns none, the message was not able to be analyzed correctly

                                 return "I couldn't find what you asked for. Please try rephrasing! Be sure to include the gender, stroke, and distance in your message."


                         try:

                                response = requests.request('GET', url, headers= header, params = querystring) # run the request

                                data = response.json()  # Format as json

                                swimmer = data['results'][0]  # Get fastest swimmer (first in json)

                                nm = swimmer['swimmer']['display_name']       # Athlete name

                                tm = swimmer['eventtime']                     # Athlete time

                                sch = swimmer['team']['abbr']                 # Athlete school

                                cod = swimmer['team']['teamcode']             # School abbreviation

                                mt = swimmer['meet']['name']                  # Meet where time achieved

                                da = swimmer['dateofswim']                    # Date when time achieved


                                conv_tm = str(dt.timedelta(seconds= float(tm)))[2:10] # Convert raw time to format hh:mm:ss.dd


                                 print(f"Swimmer found: {nm}")


                                # Randomly choose from a phrase below to simulate natural language

                                success_phrases = ["Found it!", "Coming right up!", "Got it!", "Booyah!", "Here you go!", "Check it out!"]

                                phrase = random.choice(success_phrases)


                                # Return the fully-formatted response with relevant information

                                 return f"{phrase} You're looking for {nm} from {sch} ({cod}). They went a time of {conv_tm} at the meet {mt} on {da}!"


                         except:

                                    # If the request fails, return a string explaining the error

                                 print("Something went wrong with the HTTP request.", response.status_code)

                                 return f"There was an API error (Code: {response.status_code}). Please try again later."


                # Main:

                if self.user in message.mentions:
                       # Bot only replies when mentioned in the message

                        msg = message.clean_content  # Get nicely-formatted message content

                        auth = message.author.display_name  # Find who sent the message

                        tm = str(message.created_at)  # Find time message was sent

                         print("\nMSG REC FROM " + auth + " at " + tm + " | " + msg) # Print useful debug info to console


                        # Clean up and tokenize message

                        msg = msg.lower()    # lower case

                        #                         ######################################### Remove symbols from message

                        '''Code from: https://www.geeksforgeeks.org/python-remove-punctuation-from-string/'''

                        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

                         for ele in msg:

                                 if ele in punc:

                                        msg = msg.replace(ele, "")

                        ##########################################

                        sp_msg = msg.split()    # Tokenize


                         if checkforhelp(sp_msg): # If "help" is in the message, send the help string

                                 print("User asked for help.")

                                bot_response = "HOW TO USE:\nI can tell you the fastest NCAA swimmer in any event using live data from SwimCloud. To find the fastest swimmer, tag me and ask something like \"Who is the fastest female 50 freestyler?\"\nNote: If your message contains the word \"help\", you will continue to receive this response."

                         else: # otherwise start to analyze message to find what they are asking for

                                bot_response = getSwimData(sp_msg)


                         await message.reply(bot_response.format(message)) # Send a message back to the user



                else:
                       # If bot is not mentioned in the message, do nothing

                         if message.author != self.user:

                                 print("Message received, not directed at bot.")

                         return


client = MyClient()  # Set client to the block written above and activate the bot

client.run(TOKEN)
