#############################################
#################           #################
#################   p005    #################
#################           #################
#############################################

import os

import logging
#debug, warning, info, critical, error
logging.basicConfig(
    filename="app.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,
)

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
import langchain
langchain.debug=False
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

# from redundant_filter_retriever import RedundantFilterRetriever

from dotenv import load_dotenv
load_dotenv()

chat = ChatOpenAI(temperature=0.0)
llm = llm = OpenAI(
    openai_api_key = os.getenv("OPENAI_API_KEY")
)

embeddings = OpenAIEmbeddings()

# read from chromadb only
db = Chroma(
    persist_directory = "embeddings",
    embedding_function = embeddings
)


# retriever = RedundantFilterRetriever(
#     embeddings = embeddings,
#     chroma=db
# )

### retriever has method get_relevant_documents()
### takes a string
### returns a list of documents

retriever = db.as_retriever()
memory = ConversationBufferMemory()# Set up memory to maintain context
chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type ="stuff",
    memory=memory,
)

# conversational_chain = ConversationChain(llm=chat, memory=memory, retriever=retriever)# Create a conversational chain

### not using this  ###
def run_similarity_search(query):
    results = db.similarity_search(x)
    for result in results:
        print("\n", result.page_content)

logging.info("start process")

t=True
while t:
    user_input = input("your query --> ")
    if user_input == 'q' or user_input  == 'Q':
        break

    logging.info(f"user --> {user_input}")
    
    response = chain.invoke(user_input)
    # if response['history'] != None:
    #     logging.info(f".bot --> {response["history"]}")

    # run_similarity_search(user_input)
    ### response = conversational_chain.run(user_input)
    result = response['result']
    print(f".... bot.. --> {result}")
    logging.info(f".bot --> {result}")



