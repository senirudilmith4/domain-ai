from ingestion.load_docs import load_documents

docs = load_documents()
print(f"Loaded {len(docs)} documents")
print(docs[0][:300])  # preview
