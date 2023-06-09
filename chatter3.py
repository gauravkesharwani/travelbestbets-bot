import os
from dotenv import load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.agents import Tool, load_tools, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.utilities import GoogleSearchAPIWrapper, OpenWeatherMapAPIWrapper

load_dotenv()

weather = OpenWeatherMapAPIWrapper()
CHATGPT_MODEL = os.environ.get("CHATGPT_MODEL")
search = GoogleSearchAPIWrapper()


def search_tbb(query):
    query = f'travelbestbets.com {query}'
    result_text = search.run(query)
    result_link = search.results(query, 3)[0]['link']
    result_link2 = search.results(query, 3)[1]['link']
    result_link3 = search.results(query, 3)[2]['link']

    return f'{result_text} source:{result_link} {result_link2} {result_link3}'


def greeter(query):
    return query


tools = [
    Tool(
        name="Greeter",
        func=greeter,
        description="useful when user greets and non travel related queries.",
        return_direct=True

    ),
    Tool(
        name="TravelBestBets",
        func=search_tbb,
        description="useful for when you need to answer questions about travel packages and deals. Provide full query into the tool. Provide source link with answer.",
        return_direct=True

    ),
    Tool(
        name="Google",
        func=search.run,
        description="useful when you need to answer any other question.Do not use for travel deal related questions",
        return_direct=True

    )

]

tools.extend(load_tools(["openweathermap-api"]))

llm = ChatOpenAI(model=CHATGPT_MODEL)

agent = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

template = """You are a bot travel agents for travelbestbets called TravelBot.
Always answer the questions from only the context below with itinerary and pricing information. 
Do not make up any answer
If you don't have the answer , say 'I don't know'

Include source link in inside 'a' tag
Change new line character in response to <br>

Context:
{context}


Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    # input_variables=["chat_history", "human_input", "context"],
    input_variables=["human_input", "context"],
    template=template
)
memory = ConversationBufferWindowMemory(memory_key="chat_history", input_key="human_input", k=2)
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    # memory=memory,
)


def process_response(response):
    if "I don't" in response:
        print('found i dont know')
        return '''I can't find a deal but one of our travel consultants would be happy to help you.<br> To get a 
        quote click here: https://travelbestbets.com/request-a-quote/ <br> Or feel free to contact our office: <br> ☎ 
        1-877-523-7823 <br> 📧 info@travelbestbets.com <br> And get our amazing deals sent right to your inbox. Sign 
        up for our weekly Travel Best Bets Newsletter here: https://travelbestbets.com/services/best-bets-newsletter/ 
        '''
    else:
        return response


def get_response(query):
    global llm_chain

    try:

        agent_response = agent.run(input=query)

        response = llm_chain.run({'context': agent_response, 'human_input': query})

    except Exception as e:
        print(e)
        return "Unable to complete request."

    print(response)

    return process_response(response)


def reset():
    global memory
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=2)
