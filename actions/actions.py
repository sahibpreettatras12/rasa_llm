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
from rasa_sdk.events import UserUtteranceReverted, SlotSet,ActionExecuted,AllSlotsReset
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


answer_iter=0

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

propmt2 = f"""Given you are Artificial intelligent system now recommend 1 tough interview questions 
for each element in {result} and return the output in the json format below

topic1 : [list of questions for topic1]

and so on

"""
answer_list=[]
result = get_completion(prompt=propmt2,temperature=0.0)
result = ast.literal_eval(result)
print("result type is  -->",type(result))

class ActionResultCheck(Action):

    def name(self) -> Text:
        return "action_result_check"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        global answer_iter

        if answer_iter >= 4:
            print(answer_list)
            dispatcher.utter_message(text='You have got good marks')
            return []
        elif answer_iter ==0:
            dispatcher.utter_message(text="Hi This is a hirebot happy to help")
            # dispatcher.utter_message(response='utter_result_check')
            
            p = tracker.get_slot('answer_1')
            print('we got ',p)
            print('Till now we have',answer_list)
            return []

        else:
            dispatcher.utter_message(text='Please complete the assignment')
            return []
        # print(result)
        # for i,j in result.items():
        #     temp = 0
        #     for question in j:
        #             if temp==0:
        #                 dispatcher.utter_message(text=f"""For **{i}** you have """)
        #                 dispatcher.utter_message(text=f"""1.  **{question}** """)
                        
        #             else:
        #                 dispatcher.utter_message(text=f"""1.  **{question}** """)

                    
        #             #just to check which we are on
        #             temp+=1

        

class ActionAskAnswer1(Action):

    def name(self) -> Text:
        return "action_ask_answer_1"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        global answer_iter

        if answer_iter==0:
            dispatcher.utter_message(text=f"We are Good Let's get educated together")

        # just fetching only first topic's questions because we want to
        # show only first topic's questions to user 

        print("Length of dictionary is ",len(result))

        if answer_iter in [0,1,2,3,4]:
            first_key = next(iter(result))
            answer_iter+=1
            first_value = result[first_key]

            #removing key from dictionary (Questions we already asked)
            del result[first_key]

            new_result = {first_key:first_value} 
            
            for i,j in new_result.items():
                temp = 0

                # just want to show only one question per topic because 
                # prompts can give us more than one questions per topic
                print('J was -->',j)
                
                if len(j)>=1:
                    j = j[:1]
                    for question in j:
                            # if temp==0:
                            dispatcher.utter_message(text=f"""For **{i}** you have """)
                            dispatcher.utter_message(text=f"""1.  **{question}** """)
                                
                            # else:
                            #     dispatcher.utter_message(text=f"""1.  **{question}** """)
                            
                            # temp+=1
                else:
                    dispatcher.utter_message(text='Marks the end of assignment')
            return []
        else:
            dispatcher.utter_message(text='Assignment is ended here how can i assist further')

class ActionValidateAnswer1(Action):

    def name(self) -> Text:
        return "action_validate_answer_1"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # update the number of answers we have answered
        print("Answer Iter  -->",answer_iter)
        if answer_iter in [0,1,2,3,4]:
            dispatcher.utter_message(text=f"We have saved your response for question {int(answer_iter)}")
            # global answer_iter
            # answer_iter+=1
            
            # dispatcher.utter_message(text=f"I have fetched your answer thanks")

            p = tracker.get_slot('answer_1')
            dispatcher.utter_message(text = f'we got **{p}** ')
            answer_list.append(p)

            return [AllSlotsReset()]
        
        else:
            dispatcher.utter_message(text=f"""
                                      You have submitted atleast 4 answers
                                     \n so you can check scoes""")
            return []

    
        # print('Answer 1 is  -->',tracker.get_slot('answer_1'))
        # if tracker.get_slot('answer_1'):
        #     dispatcher.utter_message(text=f"We have saved your response")
        #     return []
        # else:
        #     dispatcher.utter_message(text=f"I have fetched your answer thankies")

        #     p = tracker.get_slot('answer_1')
        #     dispatcher.utter_message(text = f'we got **{p}** ')
        #     answer_list.append(p)

        #     return []