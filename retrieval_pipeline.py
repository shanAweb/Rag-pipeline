from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persistent_directory = "db/chromedb"

embedding_model = GoogleGenerativeAIEmbeddings(model= "gemini-embedding-001")

db = Chroma(persist_directory=persistent_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space" : "cosine"})

query = input("Enter your query... ")

retriever = db.as_retriever(search_kwargs={"k" : 3})

relevant_docs = retriever.invoke(query)

print(f"User query: {query}")
print("-----Context-----")
# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}: \n {doc.page_content}\n")

combined_input = f"""Based on the following document, please answer this question {query} 
Documents: 
{chr(10).join([f"{doc.page_content}" for doc in relevant_docs])}"""

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input)
]
result = model.invoke(messages)

print("---- Generated Response ----")
print("Content only: ")
print(result.content)