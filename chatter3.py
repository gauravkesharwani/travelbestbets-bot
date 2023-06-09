from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, load_tools
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.utilities import GoogleSearchAPIWrapper, OpenWeatherMapAPIWrapper
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

weather = OpenWeatherMapAPIWrapper()


def search_tbb(query):
    query = f'travelbestbets.com {query}'
    result_text = search.run(query)
    result_link = search.results(query, 1)[0]['link']
    return f'{result_text} {result_link}'


template = """
You are a bot travel agents for travelbestbets called TravelBot.
Always Answer the following questions with itinerary and pricing information.
If you don't have the answer , say 'I don't know'
Include source link in inside 'a' tag

{chat_history}

human: {input}:
"""

prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template=template
)
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=2)

search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name="TravelBestBets",
        func=search_tbb,
        description="useful for when you need to answer questions about travel packages and deals. Provide source link with answer.",

    ),
    Tool(
        name="Google",
        func=search.run,
        description="useful when you need to answer any other question.Do not use for travel deal related questions",

    )

]

tools.extend(load_tools(["openweathermap-api"]))

prefix = """You are a bot travel agents for travelbestbets called TravelBot.
Always Answer the following questions with itinerary and pricing information. 
If you don't have the answer , say 'I don't know'
Include source link in inside 'a' tag
You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"]
)

llm = ChatOpenAI(model='gpt-4')

llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True, max_iterations=1)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)


def process_response(response):
    if 'output' in response:

        answer = response['output']

        if "I don't know" in answer:
            return '''I can't find a deal but one of our travel consultants would be happy to help you.<br> To get a 
               quote click here: <a href="https://travelbestbets.com/request-a-quote/">Request a quote</a> <br> Or feel free to contact our office: <br> ☎ 
               1-877-523-7823 <br> 📧 info@travelbestbets.com <br> And get our amazing deals sent right to your inbox. Sign 
               up for our weekly Travel Best Bets Newsletter here: <a href="https://travelbestbets.com/services/best-bets-newsletter/">Newsletter</a> 
               '''

        return answer
    if "I don't know" in response:
        return '''I can't find a deal but one of our travel consultants would be happy to help you.<br> To get a 
        quote click here: https://travelbestbets.com/request-a-quote/ <br> Or feel free to contact our office: <br> ☎ 
        1-877-523-7823 <br> 📧 info@travelbestbets.com <br> And get our amazing deals sent right to your inbox. Sign 
        up for our weekly Travel Best Bets Newsletter here: https://travelbestbets.com/services/best-bets-newsletter/ 
        '''
    else:
        return response


def get_response(query):
    global agent_chain

    try:
        response = agent_chain.run(query)

    except Exception as e:
        print(e)
        return "Unable to complete request. Please retry."

    print(response)

    return process_response(response)


def reset():
    global memory
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=2)
