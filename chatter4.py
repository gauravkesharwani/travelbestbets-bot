import os
from langchain.agents import Tool, load_tools, initialize_agent, AgentType
from langchain.chains.question_answering import load_qa_chain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.utilities import GoogleSearchAPIWrapper, OpenWeatherMapAPIWrapper, GoogleSerperAPIWrapper
from llama_index import download_loader
from dotenv import load_dotenv

load_dotenv()

CHATGPT_MODEL = os.environ.get("CHATGPT_MODEL")

SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
loader = SimpleDirectoryReader('./data', recursive=True, exclude_hidden=True)
documents = loader.load_langchain_documents()

weather = OpenWeatherMapAPIWrapper()
google = GoogleSearchAPIWrapper()
serper = GoogleSerperAPIWrapper()


def search_google_with_source(url, query):
    if url == 'travelbestbets.com':
        result_link = google.results(f'{url} + {query}', 1)[0]['link']
    else:
        result_link = url

    search_term = f'{url} + {query}'
    print(search_term)

    result_text = google.run(search_term)
    # result_text = ddg.run(search_term)

    response = f'{result_text} source:{result_link}'
    print(response)
    return response


def search_serper_with_source(url, query):
    search_term = f'{url} + {query}'

    print(f'Searching Serper:{search_term}')
    result_text = serper.run(search_term)
    result_link = serper.results(search_term)['organic'][0]['link']

    if 'travelbestbets.com' not in result_link:
        result_link = 'http://www.xyz.com'

    response = f'{result_text} source:{result_link}'

    print(response)
    return response


llm = ChatOpenAI(temperature=0, model=CHATGPT_MODEL)

prompt_url_lookup = """Provide url for any trip , deal, package tour related question to any destination from context below only.
Do not make up any answer.
Answer truthfully. 
If context doesn't have the answer just say that 'I don't know'.

Context:
{context}

Question: what are some good longstay trips to Europe
Answer : travelbestbets.com/special-interest-trips/longstay-holidays/
Question: {question}
Answer: 

"""

PROMPT_LOOKUP = PromptTemplate(
    template=prompt_url_lookup, input_variables=["context", "question"]
)
chain_lookup = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT_LOOKUP)

prompt_tbb_deal = """You are a bot travel agents for travelbestbets called TravelBot.
Always answer the questions regarding travel deals, flight deals and packages from only the context below with itinerary and pricing information. 
Do not make up any answer.
If you don't have the answer in the context say 'I don't know'
Include source link in inside <a> tag with target="_blank".
Do not provide any other email other than info@travelbestbets.com 
Do not provide any other link other than from travelbestbets.
Italy & Greek isles cruise is part of mediterranean cruise
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
    llm=llm,#ChatOpenAI(temperature=0, model='gpt-3.5-turbo'),
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
    prompt=PROMPT_SEARCH_GOOGLE,
    verbose=True

)


def check_words_in_string(string):
    words_to_check = ["dont", "sorry", "don't"]
    for word in words_to_check:
        if word in string:
            return True
    return False


def search_tbb(query):
    if "mediterranean" in query.lower() and "cruise" in query.lower():
        query = f'{query} and greek isles'

    url = chain_lookup({"input_documents": documents, "question": query}, return_only_outputs=True)['output_text']

    print(f'lookup result: {url}')

    if check_words_in_string(url):
        url = 'travelbestbets.com'

    print(url)

    deal_info = search_serper_with_source(url, query)

    fa = chain_tbb_deal({"context": deal_info, "question": query})['text']
    return fa


def search_google(query):
    result_text = serper.run(query)
    fa = chain_search_google({"context": result_text, "question": query})['text']
    return fa


prompt_greeter = """You are a AI travel agents bot for travelbestbets.
Your name is TravelBot.
You never say you are an AI language model.
Answer as human travel agent 

Question: {question}
Answer:
"""

PROMPT_GREETER = PromptTemplate(
    template=prompt_greeter, input_variables=["question"]
)


def greeter(query):
    chain = LLMChain(llm=llm, prompt=PROMPT_GREETER)
    return chain.run(query)


tools = [
    Tool(
        name="Greeter",
        func=greeter,
        description="useful when user greets or say hi or hello and non travel related queries.",
        return_direct=True
    ),
    Tool(
        name="TravelBestBets",
        func=search_tbb,
        description="useful for when you need to answer questions about travel packages and deals. Pass full query into the tool. Provide source link with answer.",
        return_direct=True

    ),
    Tool(
        name="Google",
        func=search_google,
        description="useful when you need to answer any other travel related question.Do not use for travel deal related questions",
        return_direct=True

    )
]

tools.extend(load_tools(["openweathermap-api"]))
agent = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


def process_response(response):
    if check_words_in_string(response):
        print('found i dont know')
        return '''I can't find a deal but one of our travel consultants would be happy to help you.<br> To get a 
        quote click here: <a href="https://travelbestbets.com/request-a-quote/" target="_blank">Request a quote</a> <br> Or feel free to contact our office: <br> ☎ 
        1-877-523-7823 <br> 📧 info@travelbestbets.com <br> And get our amazing deals sent right to your inbox. Sign 
        up for our weekly Travel Best Bets Newsletter here: <a href="https://travelbestbets.com/services/best-bets-newsletter/" target="_blank">Newsletter</a>
        '''
    elif 'xyz.com' in response:
        return response.replace('<a href="http://www.xyz.com" target="_blank">source</a>', """<br>One of our travel consultants would be happy to help you.<br> To get a 
        quote click here: <a href="https://travelbestbets.com/request-a-quote/" target="_blank">Request a quote</a>  <br> Or feel free to contact our office: <br> ☎ 
        1-877-523-7823 <br> 📧 info@travelbestbets.com """)
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

# def reset():
# global memory
# memory = ConversationBufferWindowMemory(memory_key="chat_history", k=2)
