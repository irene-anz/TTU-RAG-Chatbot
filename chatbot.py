from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

print("Step 1: Loading embeddings...")
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Step 2: Embeddings loaded!")

print("Step 3: Loading vector database...")
db = Chroma(persist_directory="vectordb", embedding_function=embedding)
print("Step 4: Vector database loaded!")

print("Step 5: Loading LLM...")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
print("Step 6: LLM loaded!")

print("Step 7: Building retriever...")
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 20}
)
print("Step 8: Retriever ready!")

print("Step 9: Building chain...")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_context(input_dict):
    question = input_dict.get("standalone_question", input_dict.get("input", ""))
    docs = retriever.invoke(question)
    return format_docs(docs)

def condense_question(input_dict):
    chat_history = input_dict.get("chat_history", [])
    question = input_dict.get("input", "")
    
    if not chat_history:
        return question
    
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system", """Given the chat history and the latest user question about Tatung University (TTU),
        rephrase it as a standalone question.
        TTU and Tatung University refer to the same institution in Taipei, Taiwan.
        Always expand TTU to Tatung University in the rephrased question.
        Return ONLY the rephrased question, nothing else."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    condense_chain = condense_prompt | llm | StrOutputParser()
    return condense_chain.invoke({
        "chat_history": chat_history,
        "input": question
    })

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant for Tatung University, also known as TTU, located in Taipei, Taiwan.
TTU and Tatung University refer to the same institution.
Use the context below to answer the question as clearly and completely as possible.
If the context contains partial information, use it and supplement with what you know about TTU.
If the context has no relevant information at all, say so honestly.

Context:
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

def run_chain(input_dict):
    standalone = condense_question(input_dict)
    context = get_context({"input": standalone})
    
    prompt_input = {
        "context": context,
        "chat_history": input_dict.get("chat_history", []),
        "input": input_dict.get("input", "")
    }
    
    response = (answer_prompt | llm | StrOutputParser()).invoke(prompt_input)
    return {"answer": response}

chain = RunnableLambda(run_chain)
print("Step 10: Chain ready!")

if __name__ == "__main__":
    print("Chatbot ready! Type 'exit' to quit.\n")
    chat_history = []

    while True:
        question = input("Ask: ")

        if question.strip().lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        if not question.strip():
            continue

        response = chain.invoke({
            "input": question,
            "chat_history": chat_history
        })

        answer = response["answer"]
        print(f"\nAnswer: {answer}\n")

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))

        if len(chat_history) > 20:
            chat_history = chat_history[-20:]