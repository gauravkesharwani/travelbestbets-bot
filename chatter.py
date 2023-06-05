import os
import pinecone
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.tools import Tool
from langchain.utilities import OpenWeatherMapAPIWrapper
from langchain.agents import initialize_agent, AgentType

load_dotenv()

os.environ["OPENWEATHERMAP_API_KEY"] = os.getenv('OPENWEATHERMAP_API_KEY')

weather = OpenWeatherMapAPIWrapper()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
PINECONE_INDEX = os.getenv('PINECONE_INDEX')
CORE_DB = os.getenv('CORE_DB')
TBB_DB = os.getenv('TBB_DB')

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(PINECONE_INDEX)
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone(index, embeddings.embed_query, "text")

prompt_template1 = """You are friendly helpful bot for travel company called travelbestbets.
Your name is TravelBot
You never say you're a machine or an AI language model
Answer generic user travel related queries from your knowledgebase and context below to arrive at the best answer. 
For package and deal queries , answer from travelbestbets source with pricing and date information. Also provide a link from travelbestbets site.
For realtime information and latest deals , answer from travelbestbets source with pricing and date information. Also provide a link from travelbestbets site.
Do not provide link from any other website apart from travelbestbets
For any weather related queries , just say 'Providing dummy weather info'
Change new line character in response to <br>
Enclose url in the url inside href tag

{summaries}

{chat_history}
Question: {question}
Answer:"""
PROMPT1 = PromptTemplate(
    template=prompt_template1, input_variables=["summaries", "chat_history", "question"]
)

docsearch_Travel = Pinecone.from_existing_index(PINECONE_INDEX, embeddings, namespace=CORE_DB)
docsearch_BestBet = Pinecone.from_existing_index(PINECONE_INDEX, embeddings, namespace=TBB_DB)

chain = load_qa_with_sources_chain(
    ChatOpenAI(model="gpt-4"),
    chain_type="stuff",
    verbose=True,
    prompt=PROMPT1,
    memory=ConversationBufferWindowMemory(
        k=3,
        memory_key="chat_history",
        input_key="question"
    )
)

tools = [
    Tool.from_function(
        func=weather.run,
        name="weather",
        description="useful for when you need to answer questions about weather"
        # coroutine= ... <- you can specify an async method if desired as well
    ),
]

agent_chain = initialize_agent(
    tools=tools,
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


def process_response(response, query):
    if "dummy" in response:
        return agent_chain.run(query)
    return response


def get_response(query):
    global chain
    docs = []
    try:
        docs_tbb = docsearch_BestBet.similarity_search(query, k=2)
        docs.extend(docs_tbb)
        docs_travel = docsearch_Travel.similarity_search(query, k=2)
        docs.extend(docs_travel)

        response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    except Exception as e:
        print(e)
        return "Unable to complete request. Please try after sometime."

    print(response)

    return process_response(response['output_text'],query)


def reset():
    global chain
    chain = load_qa_with_sources_chain(
        ChatOpenAI(model="gpt-4"),
        chain_type="stuff",
        verbose=True,
        prompt=PROMPT1,
        memory=ConversationBufferWindowMemory(
            k=3,
            memory_key="chat_history",
            input_key="question"
        )
    )
