import os


from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


print("Initializing...")

embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
llm = ChatOpenAI()

vectorestore = PineconeVectorStore(
    index_name=os.environ.get("PINECONE_INDEX_NAME"), embedding=embeddings
)

retriever = vectorestore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that accurately answers questions based on the provided context.",
        ),
        ("human", "{context}\n\nQuestion: {question}\nAnswer:"),
    ]
)


def create_retrival_chain():
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )  # ← this is a chain that first retrieves relevant documents and then passes them to the LLM for answering

    return retrieval_chain


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


if __name__ == "__main__":
    print("Asking questions...")

    question = "What is Pinecone in machine learning"

    print("\n" + "=" * 50 + "\n")
    print("IMPLEMENTATION 1: Using raw LLM invocation (No RAG)\n")
    result_raw = llm.invoke([HumanMessage(content=question)])
    print("Answer:", result_raw.content)

    print("\n" + "=" * 50 + "\n")
    print("IMPLEMENTATION 2: Implementation with RAG and LCRL\n")
    chain_with_rag = create_retrival_chain()
    result = chain_with_rag.invoke({"question": question})
    print("Answer:", result)
