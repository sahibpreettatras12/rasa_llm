# This files contains your custom actions which can be used to run
# custom Python code.

# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

# This is a simple example for a custom action which utters "Hello World!"
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from typing import Any, Text, Dict, List
from dotenv import load_dotenv,find_dotenv
from sentence_transformers import SentenceTransformer,util

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
                                               vectordb.as_retriever(search_kwargs={"k": 1}), memory=memory)
query1 = """list down all the technologies on which the person has worked in their resume and give out response
 in python list"""
result = pdf_qa({"question": query1})
result  = result['answer']


result = f""" given you are the samrt assistant now given {result} list down all the technologies on which 
a person has worked give out response in python list
element1 : a technology 
so on
Format the output as Python List 

with different elements where each element is a technology 

"""

def find_list_indices(string):
    list_indices = []
    stack = []

    for i, char in enumerate(string):
        if char == '[':
            stack.append(i)
        elif char == ']':
            if stack:
                start_index = stack.pop()
                list_indices.append((start_index, i))

    return list_indices


result = get_completion(prompt=result,temperature=0.0)


indices = find_list_indices(result)
print(indices)

# just to be sure we are fetching list used --> find_list_indices function
# then fetched whatever first list we found from result
result = result[indices[0][0]:indices[0][1]+1]

print('Before ast',result)
result = ast.literal_eval(result)
print('After ast',result)


result = [element.lower() for element in result]
print(result)


result = random.sample(result,5)


# propmt2 = f"""Given you are Artificial intelligent system now recommend 1 tough interview questions 
# for each element in {result} and return the output in the json format below

# topic1 : [list of questions for topic1]

# and so on

# """

propmt2 = f"""Given you are Artificial intelligent system now recommend 1 tough interview question and answer
for each element in {result} and return the output in the json format below

topic1 : [list of questions for topic1,answer of questions for topic1]

and so on

"""

correct_answer_list = []
answer_list=[]
result = get_completion(prompt=propmt2,temperature=0.0)
result = ast.literal_eval(result)
print("result type is  -->",type(result))

class ActionWelcomeOnBoard(Action):

    def name(self) -> Text:
        return "action_welcome_on_board"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            global correct_answer_list

            if len(correct_answer_list)==0:
                buttons = []

                    #append the response of API in the form of title and payload

                buttons.append({"title": 'START QUIZ' , 'payload': '/start_qna'})

                dispatcher.utter_message(text="""Welcome On board I am Interview Bot""")
                dispatcher.utter_message(text="""I can help you get some Training on Topics of your RESUME""")
                dispatcher.utter_message(text="""To get started""")
                dispatcher.utter_message(text="""Hit on START or type in Let's start""",buttons=buttons)

                return []
            
            else:
                buttons = []

        #append the response of API in the form of title and payload

                buttons.append({"title": 'Continue Quiz' , 'payload': '/start_qna'})

                dispatcher.utter_message(text='We are also now inculcating Video Response for Quiz',buttons=buttons)




class ActionResultCheck(Action):

    def name(self) -> Text:
        return "action_result_check"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        global answer_iter
        model = SentenceTransformer('all-MiniLM-L6-v2')
        buttons = []

        #append the response of API in the form of title and payload

        buttons.append({"title": 'Next' , 'payload': '/start_qna'})
        
        if answer_iter >= 4:

            final_score=[]

            #Compute embedding for both lists
            embeddings1 = model.encode(answer_list, convert_to_tensor=False)
            embeddings2 = model.encode(correct_answer_list, convert_to_tensor=False)


            #Compute cosine-similarities
            cosine_scores = util.cos_sim(embeddings1, embeddings2)

            #Output the pairs with their score
            for i in range(len(answer_list)):
                k=cosine_scores[i][i]
                final_score.append(k*2)

            dispatcher.utter_message(text=f'You have {round(sum(final_score).item(),1)} out of 10')
            return []
        elif answer_iter ==0:
            dispatcher.utter_message(text="""Hi This is a hirebot happy to help. And to check your scores.
                                            Please start the quiz    """)
            # dispatcher.utter_message(response='utter_result_check')
            
            p = tracker.get_slot('answer_1')
            print('we got ',p)
            print('Till now we have',answer_list)
            return []

        else:
            dispatcher.utter_message(text='Please complete the assignment',buttons=buttons)
            return []


        

class ActionAskAnswer1(Action):

    def name(self) -> Text:
        return "action_ask_answer_1"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        global answer_iter
        global correct_answer_list
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
                

                #then display it using dispatcher
                if len(j)>=2:
                    answer_per_topic = j[1]
                    # j = random.sample(j,1)
                    j = j[:1]
                    
                    correct_answer_list.append(answer_per_topic)
                    for question in j:

                            dispatcher.utter_message(text=f"""For **{i}** you have """)
                            dispatcher.utter_message(text=f"""1.  **{question}** """)

                elif len(j)==1:
                    answer_per_topic = 'Sorry we Have No answer'
                    
                    correct_answer_list.append(answer_per_topic)
                    j = random.sample(j,1)
                    j = j[:1]
                    for question in j:

                            dispatcher.utter_message(text=f"""For **{i}** you have """)
                            dispatcher.utter_message(text=f"""1.  **{question}** """)
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
        buttons = []

                #append the response of API in the form of title and payload

        buttons.append({"title": 'Next' , 'payload': '/start_qna'})
        
        # update the number of answers we have answered
        print("Answer Iter  -->",answer_iter)
        if answer_iter in [0,1,2,3,4]:
            dispatcher.utter_message(text=f"We have saved your response for question {int(answer_iter)}")

            p = tracker.get_slot('answer_1')
            dispatcher.utter_message(text = f'we got **{p}** ',buttons=buttons)
            answer_list.append(p)

            return [AllSlotsReset()]
        
        else:
            buttons = []

                #append the response of API in the form of title and payload

            buttons.append({"title": 'Scores' , 'payload': '/result_check'})
            dispatcher.utter_message(text=f"""
                                      You have submitted atleast 4 answers """)
            dispatcher.utter_message(text = f"""so you can check scoes""",buttons=buttons)
            return []

