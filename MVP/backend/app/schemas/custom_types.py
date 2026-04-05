from pydantic import BaseModel

class RAGChunkAndSrc(BaseModel):  # A chunk of text + where it came from 
 ''' During Ingestion
     PDF → Split into chunks → Attach source ID → Store in vector DB '''
 chunks: list[str]
 metadatas: list[dict]


class RAGUpsertResult(BaseModel): # ex : 128 chunks were successfully stored.
    ingested : int

class RAGSearchResult(BaseModel): # Results of vector similarity search
   contexts : list[str]
   sources  : list[str]

class RAGQueryResult(BaseModel):  # Final structured RAG output (internal)
    answer: str
    sources: list[str]
    num_contexts : int