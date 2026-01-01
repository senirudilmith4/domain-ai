"""
Simple Command-Line Chatbot for RAG System
Run: python chatbot.py
"""

import sys
from pathlib import Path
from MVP.backend.pipeline.rag_pipeline import RAGPipeline
from MVP.backend.utils.logger import get_logger

logger = get_logger(__name__)


class RAGChatbot:
    """Simple CLI chatbot interface for RAG system."""
    
    def __init__(self, collection_name: str = "documents"):
        """Initialize the chatbot."""
        print("🤖 Initializing RAG Chatbot...")
        self.rag = RAGPipeline(collection_name=collection_name)
        self.conversation_history = []
        
        # Check if database has documents
        stats = self.rag.vector_store.get_collection_stats()
        
        if stats['total_documents'] == 0:
            print("\n⚠️  No documents found in database!")
            print("Would you like to ingest documents now? (yes/no): ", end="")
            
            response = input().strip().lower()
            if response in ['yes', 'y']:
                self.ingest_documents()
            else:
                print("\n❌ Cannot run chatbot without documents.")
                print("Run this first: python test_rag.py")
                sys.exit(1)
        else:
            print(f"✅ Database loaded: {stats['total_documents']} chunks available\n")
    
    def ingest_documents(self):
        """Ingest documents into the database."""
        print("\n📚 Starting document ingestion...")
        print("This may take a few minutes...\n")
        
        result = self.rag.ingest_documents(
            chunk_size=500,
            overlap=100,
            batch_size=32
        )
        
        if result['success']:
            print(f"\n✅ Ingestion complete!")
            print(f"   Documents loaded: {result['documents_loaded']}")
            print(f"   Chunks created: {result['chunks_created']}")
            print(f"   Chunks stored: {result['chunks_stored']}\n")
        else:
            print(f"\n❌ Ingestion failed: {result['message']}")
            sys.exit(1)
    
    def print_banner(self):
        """Print welcome banner."""
        print("\n" + "="*70)
        print("🤖  RAG CHATBOT  🤖")
        print("="*70)
        print("\nAsk me anything about your documents!")
        print("\nCommands:")
        print("  • Type your question and press Enter")
        print("  • 'help' - Show available commands")
        print("  • 'stats' - Show database statistics")
        print("  • 'history' - Show conversation history")
        print("  • 'clear' - Clear conversation history")
        print("  • 'quit' or 'exit' - Exit the chatbot")
        print("\n" + "-"*70 + "\n")
    
    def show_stats(self):
        """Show database statistics."""
        stats = self.rag.vector_store.get_collection_stats()
        print(f"\n📊 Database Statistics:")
        print(f"   Collection: {stats['collection_name']}")
        print(f"   Total chunks: {stats['total_documents']}")
        print(f"   Location: {stats['persist_directory']}")
        print()
    
    def show_history(self):
        """Show conversation history."""
        if not self.conversation_history:
            print("\n📝 No conversation history yet.\n")
            return
        
        print("\n📝 Conversation History:")
        print("-"*70)
        for i, entry in enumerate(self.conversation_history, 1):
            print(f"\n[{i}] Q: {entry['question']}")
            print(f"    A: {entry['answer'][:200]}{'...' if len(entry['answer']) > 200 else ''}")
        print("-"*70 + "\n")
    
    def show_help(self):
        """Show help information."""
        print("\n📖 Help:")
        print("   • Ask natural language questions about your documents")
        print("   • The bot will search relevant chunks and generate answers")
        print("   • Answers are based ONLY on your document content")
        print("   • Type 'stats' to see database info")
        print("   • Type 'history' to see past questions")
        print("   • Type 'clear' to reset conversation")
        print("   • Type 'quit' to exit\n")
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("\n🗑️  Conversation history cleared.\n")
    
    def ask_question(self, question: str):
        """Process a question and generate answer."""
        print("\n🔍 Searching documents and generating answer...\n")
        
        result = self.rag.answer_with_llm(
            query_text=question,
            n_results=3,
            max_context_length=2000
        )
        
        if result['success']:
            answer = result['answer']
            sources = result['sources']
            
            # Display answer
            print("🤖 Answer:")
            print("-"*70)
            print(answer)
            print("-"*70)
            
            # Display sources
            if sources:
                print(f"\n📚 Sources ({len(sources)} chunks used):")
                for i, source in enumerate(sources[:3], 1):
                    print(f"\n   [{i}] Similarity: {source['similarity']:.3f}")
                    print(f"       {source['text']}")
            
            # Save to history
            self.conversation_history.append({
                'question': question,
                'answer': answer,
                'sources': len(sources)
            })
            
        else:
            print(f"\n❌ Error: {result['answer']}")
        
        print()
    
    def run(self):
        """Run the chatbot interactive loop."""
        self.print_banner()
        
        while True:
            try:
                # Get user input
                user_input = input("💬 You: ").strip()
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Handle commands
                command = user_input.lower()
                
                if command in ['quit', 'exit', 'q']:
                    print("\n👋 Thanks for chatting! Goodbye!\n")
                    break
                
                elif command == 'help':
                    self.show_help()
                    continue
                
                elif command == 'stats':
                    self.show_stats()
                    continue
                
                elif command == 'history':
                    self.show_history()
                    continue
                
                elif command == 'clear':
                    self.clear_history()
                    continue
                
                # Process as question
                self.ask_question(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!\n")
                break
            
            except Exception as e:
                logger.error(f"Error in chatbot: {e}", exc_info=True)
                print(f"\n❌ Error: {str(e)}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Chatbot")
    parser.add_argument(
        "--collection",
        default="documents",
        help="ChromaDB collection name (default: documents)"
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest documents before starting chat"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize chatbot
        chatbot = RAGChatbot(collection_name=args.collection)
        
        # Ingest if requested
        if args.ingest:
            chatbot.ingest_documents()
        
        # Run chat loop
        chatbot.run()
        
    except Exception as e:
        logger.error(f"Chatbot failed: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()