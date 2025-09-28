import faiss

class Faissindex:
    def __init__(self,vector_length):
        self.index = faiss.IndexFlatIP(vector_length)

    def create(self,vectors):
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        
    def search(self,vector_query,n):
        faiss.normalize_L2(vector_query)
        scores, indexes = self.index.search(vector_query, n)
        return scores, indexes