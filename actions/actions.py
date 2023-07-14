# This files contains your custom actions which can be used to run
# custom Python code.

# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

# This is a simple example for a custom action which utters "Hello World!"
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from typing import Any, Text, Dict, List
from dotenv import load_dotenv,find_dotenv
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import UserUtteranceReverted, SlotSet,ActionExecuted
import os,ast,random
import openai
import langchain
import shutup
shutup.please()
from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from rasa_sdk.events import FollowupAction
import time
from datetime import date

month= date.today().month
year=date.today().year

dic_month={1:'January',2:'Feburary',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August'
,9:'September',10:'October',11:'November',12:'December'}

month=dic_month[month]
present_date = str(month)+' '+str(year)

_ = load_dotenv(find_dotenv())

open_ai_key = str(os.getenv('open_ai_key'))

os.environ['OPENAI_API_KEY'] = open_ai_key

def get_completion(prompt, model="gpt-3.5-turbo",temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]



pdf_path = "/home/sahibpreet/pdf_chatbot/test.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(pages, embedding=embeddings, 
                                 persist_directory=".")
vectordb.persist()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
pdf_qa = ConversationalRetrievalChain.from_llm( OpenAI(temperature=0.0,) , 
                                               vectordb.as_retriever(), memory=memory)
query1 = """list down all the technologies on which the person has worked in their resume and give out response
 in python list"""
result = pdf_qa({"question": query1})
result  = result['answer']
result = f"""
Given you are the samrt assistant now given {result} list down all the technologies on which a person has worked
give out response in python list"""
result = get_completion(prompt=result,temperature=0.0)

if (result[0]=='[') and (result[-1]==']'):
    result = ast.literal_eval(result)

result = random.sample(result,5)

propmt2 = f"""Given you are Artificial intelligent system now recommend 2 tough multiple
choice interview questions for each
element in {result} and each question can have only 2 responses and the response should only be in 
yes/no or it can be true/false.
Also store the links from which questions were made
your response should be json

topic1 : [list of questions for topic1,answers for questions for topic1,links for questions for topic1 ]
and so on

"""
result = get_completion(prompt=propmt2,temperature=0.0)
result = ast.literal_eval(result)

class ActionGreet(Action):

    def name(self) -> Text:
        return "action_greet"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Hi This is a hirebot happy to help")
        dispatcher.utter_message(response='utter_greet_action')
        
        # print(result)
        for i,j in result.items():
            temp = 0
            for question in j:
                    if temp==0:
                        dispatcher.utter_message(text=f"""For **{i}** you have """)
                        dispatcher.utter_message(text=f"""1.  **{question}** """)
                        
                    else:
                        dispatcher.utter_message(text=f"""1.  **{question}** """)

                    
                    #just to check which we are on
                    temp+=1

        return []

