#=============================
# Router 
#=============================
def route_question(query):

    query = query.lower()

    if "google" in query:
        return "docs\\Google.txt"
    
    elif "tesla" in query:
        return "docs\\Tesla.txt"
    
    elif "spacex" in query:
        return "docs\\SpaceX.txt"
    
    elif "microsoft" in query:
        return "docs\\Microsoft.txt"
    
    elif "nvidia" in query:
        return "docs\\Nvidia.txt"