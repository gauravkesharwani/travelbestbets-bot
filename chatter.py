import os
import pinecone
from dotenv import load_dotenv
from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool, load_tools
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.utilities import OpenWeatherMapAPIWrapper
from langchain.vectorstores import Pinecone

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
NAMESPACE_TBB = os.environ.get("NAMESPACE_TBB")
NAMESPACE_CORE = os.environ.get("NAMESPACE_CORE")
RETURN_DOCS_COUNT_CORE = int(os.environ.get("RETURN_DOCS_COUNT_CORE"))
RETURN_DOCS_COUNT_TBB = int(os.environ.get("RETURN_DOCS_COUNT_TBB"))
CHAT_HISTORY_COUNT = int(os.environ.get("CHAT_HISTORY_COUNT"))
CHATGPT_MODEL = os.environ.get("CHATGPT_MODEL")


weather = OpenWeatherMapAPIWrapper()
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(PINECONE_INDEX)
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone(index, embeddings.embed_query, "text")

prompt_template_tbb = """You are friendly helpful bot for travel company called travelbestbets.
Your name is TravelBot
You never say you're a machine or an AI language model
You provide package, deals and pricing information for any destination from context below only. Provide link to the source of the answer.
If you don't have an answer from below context, Provide this response 'I don't know'
Change new line character in response to <br>
Enclose url in the url inside 'a' tag

{summaries}

{chat_history}
Question: {question}
Answer:"""
PROMPT_tbb = PromptTemplate(
    template=prompt_template_tbb, input_variables=["summaries", "chat_history", "question"]
)

prompt_template_core = """You are friendly helpful bot for travel company called travelbestbets.
Your name is TravelBot
You never say you're a machine or an AI language model
Answer generic user travel related queries from your knowledgebase and context below to arrive at the best answer. 
Do not provide any source link.
Change new line character in response to <br>

{context}

{chat_history}
Question: {question}
Answer:"""
PROMPT_core = PromptTemplate(
    template=prompt_template_core, input_variables=["context", "chat_history", "question"]
)

docsearch_Travel = Pinecone.from_existing_index(PINECONE_INDEX, embeddings, namespace=NAMESPACE_CORE)
docsearch_BestBets = Pinecone.from_existing_index(PINECONE_INDEX, embeddings, namespace=NAMESPACE_TBB)

llm = ChatOpenAI(model=CHATGPT_MODEL)
memory = ConversationBufferWindowMemory(
    k=CHAT_HISTORY_COUNT,
    memory_key="chat_history",
    input_key="question")

qa_travel = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch_Travel.as_retriever(search_kwargs={"k": RETURN_DOCS_COUNT_CORE}),

    chain_type_kwargs={
        "verbose": True,
        "prompt": PROMPT_core,
        "memory": memory
    }
)

qa_BestBets = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch_BestBets.as_retriever(search_kwargs={"k": RETURN_DOCS_COUNT_TBB}),
    chain_type_kwargs={
        "verbose": True,
        "prompt": PROMPT_tbb,
        "memory": memory
    }

)


def get_tbb(query):
    answer = qa_BestBets({"question": query}, return_only_outputs=True)['answer']
    return answer


tools = [
    Tool(
        name="Travel Best Bets",
        func=get_tbb,
        description="useful for when you need to answer questions about travel deals ,travel packages and pricing "
                    "about any location. Pass the whole question as input. Say I don't know if you dont know the "
                    "answer",
        return_direct=True
    ),
    Tool(
        name="Generic Core",
        func=qa_travel.run,
        description="useful for when you need to answer any generic travel related questions or about company yello jello",
        return_direct=True
    )

]

tools.extend(load_tools(["openweathermap-api"]))

agent = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


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
    global agent

    try:
        response = agent(query)

    except Exception as e:
        print(e)
        return "Unable to complete request. Please retry."

    print(response)

    return process_response(response)


def reset():
    global memory
    memory = ConversationBufferWindowMemory(
        k=CHAT_HISTORY_COUNT,
        memory_key="chat_history",
        input_key="question")
