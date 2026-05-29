from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

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
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given the chat history and the latest user question about Tatung University (TTU),
    rephrase it as a standalone question that can be understood without the chat history.
    Important rules:
    - TTU and Tatung University refer to the same institution in Taipei, Taiwan.
    - Always expand TTU to Tatung University in the rephrased question.
    - For example: "where is TTU located" becomes "where is Tatung University located"
    Do NOT answer the question, just rephrase it."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_prompt
)

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

question_answer_chain = create_stuff_documents_chain(llm, answer_prompt)
chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
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