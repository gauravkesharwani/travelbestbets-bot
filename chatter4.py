import os
from langchain.agents import Tool, load_tools, initialize_agent, AgentType
from langchain.chains.question_answering import load_qa_chain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.utilities import GoogleSearchAPIWrapper, OpenWeatherMapAPIWrapper
from llama_index import download_loader
from dotenv import load_dotenv

load_dotenv()


CHATGPT_MODEL = os.environ.get("CHATGPT_MODEL")


SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
loader = SimpleDirectoryReader('./data', recursive=True, exclude_hidden=True)
documents = loader.load_langchain_documents()

weather = OpenWeatherMapAPIWrapper()
google = GoogleSearchAPIWrapper()


def search_google_with_source(query):
    result_text = google.run(query)
    result_link = google.results(query, 1)[0]['link']
    return f'{result_text} source:{result_link}'


llm = ChatOpenAI(temperature=0, model=CHATGPT_MODEL)

prompt_url_lookup = """Provide url for any trip , deal related question to any destination from context below only.
Do not make up any answer. 
If you context dosent have the answer, just say that you don't know.

{context}

Question: what are some good longstay trips to Europe
Answer : https://travelbestbets.com/special-interest-trips/longstay-holidays/
Question: {question}
Answer: 

"""

PROMPT_LOOKUP = PromptTemplate(
    template=prompt_url_lookup, input_variables=["context", "question"]
)
chain_lookup = load_qa_chain(ChatOpenAI(temperature=0, model='gpt-4'), chain_type="stuff", prompt=PROMPT_LOOKUP)

prompt_tbb_deal = """You are a bot travel agents for travelbestbets called TravelBot.
Always answer the questions from only the context below with itinerary and pricing information. 
Do not make up any answer
If you don't have the answer , say 'I don't know'
Include source link in inside <a> tag with target="_blank"
Do not provide any other email other than info@travelbestbet.com 
Do not provide any other link other than from travelbestbets
Change new line character in response to <br>


Context:
{context}

Question: {question}
Answer:

"""

PROMPT_TBB_DEAL = PromptTemplate(
    template=prompt_tbb_deal, input_variables=["context", "question"]
)
chain_tbb_deal = LLMChain(
    llm=ChatOpenAI(temperature=0, model='gpt-4'),
    prompt=PROMPT_TBB_DEAL
)

prompt_search_google = """You are a bot travel agents for travelbestbets called TravelBot.
Answer question from your knowledgebase and the context provided below
Do not provide any source link 
Change new line character in response to <br>


Context:
{context}

Question: {question}
Answer:

"""

PROMPT_SEARCH_GOOGLE = PromptTemplate(
    template=prompt_search_google, input_variables=["context", "question"]
)
chain_search_google = LLMChain(
    llm=llm,
    prompt=PROMPT_SEARCH_GOOGLE
)


def search_tbb(query):
    url = chain_lookup({"input_documents": documents, "question": query}, return_only_outputs=True)['output_text']

    if "don't" in url:
        url = 'travelbestbets.com'

    print(url)

    deal_info = search_google_with_source(f'{url} {query}')
    fa = chain_tbb_deal({"context": deal_info, "question": query})['text']
    return fa


def search_google(query):
    result_text = google.run(query)
    fa = chain_search_google({"context": result_text, "question": query})['text']
    return fa


def greeter(query):
    fa = chain_search_google({"context": '', "question": query})['text']
    return fa


search_tbb('any deal to mexico')

search_google('best restaurant in milan')

greeter('hello')

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
        func=search_google,
        description="useful when you need to answer any other question.Do not use for travel deal related questions",
        return_direct=True

    )

]

tools.extend(load_tools(["openweathermap-api"]))
agent = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

def process_response(response):
    if "don't" in response:
        print('found i dont know')
        return '''I can't find a deal but one of our travel consultants would be happy to help you.<br> To get a 
        quote click here: https://travelbestbets.com/request-a-quote/ <br> Or feel free to contact our office: <br> ☎ 
        1-877-523-7823 <br> 📧 info@travelbestbets.com <br> And get our amazing deals sent right to your inbox. Sign 
        up for our weekly Travel Best Bets Newsletter here: https://travelbestbets.com/services/best-bets-newsletter/ 
        '''
    else:
        return response


def get_response(query):
    global agent

    try:

        agent_response = agent.run(query)
    except Exception as e:
        print(e)
        return "Unable to complete request."

    print(agent_response)

    return process_response(agent_response)


#def reset():
    #global memory
    #memory = ConversationBufferWindowMemory(memory_key="chat_history", k=2)
