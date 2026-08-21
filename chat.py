from retrievers.metadata import retrieve_documents
from utils.formatter import format_docs
from rag_pipeline import run_rag

# =========================
# Chat
# =========================
def chat(
    vectorstore,
    bm25, 
    documents,
    generation_chain
):

    while True:

        query = input(
            "\nYou: "
        ).strip()

        if query.lower() in [
            "/bye",
            "bye",
            "exit",
            "quit"
        ]:
            print("\nGoodbye!")
            break

        
        result = run_rag(
            vectorstore,
            bm25,
            documents,
            query
        )
        # docs = retrieve_documents(
        #     vectorstore,
        #     bm25,
        #     documents,
        #     query
        # )

        # docs = hybrid_search(
        #     vectorstore,
        #     documents,

        #     )

        # if docs is None:
        #     print("Assistant:")
        #     print("I couldn't find any relevant information.")
        #     continue

        # context = format_docs(
        #     docs
        # )

        # answer = generation_chain.invoke(
        #     {
        #         "context": context, 
        #         "question": query
        #     }
        # ) 

        print("\nAssistant:")
        print(result["answer"])


    

