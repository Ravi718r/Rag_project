#=============================
# Seprate Retrieval Strategy 
#=============================
def create_similarity_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs={
            "k":3
        }
    )