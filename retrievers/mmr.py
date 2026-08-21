#=============================
# MMR Retriever  
#=============================
def create_mmr_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type = "mmr",
        search_kwargs={
            "k":3,
            "fetch_k": 10
        }
    )
