from database.postgres import PostgresDB
from database.chroma import ChromaDB
from services.reranking import RerankerService
from services.vietnamese_llm_helper import VietnameseLLMHelper


class EnhancedSearchService:
    """
    Enhanced search service that integrates all improvements:
    - Overlapping chunks
    - Better document generation
    - Query understanding
    - Reranking
    """

    def __init__(self):
        self.postgres_db = PostgresDB().connect()
        self.chroma_db = ChromaDB().connect()
        self.vi_helper = VietnameseLLMHelper()
        self.reranker = RerankerService()

    def search(self, query, language="en", n_results=5, filters=None):
        """
        Enhanced search that uses a multi-stage process:
        1. Query understanding/translation if needed
        2. Initial retrieval with chunking
        3. Reranking of results

        Args:
            query: The search query
            language: Query language ("en" or "vi")
            n_results: Number of results to return
            filters: Optional filters for the search

        Returns:
            Reranked search results
        """
        try:
            # Step 1: Enhance/translate query if in Vietnamese
            enhanced_query = query
            if language == "vi":
                enhanced_query = self.vi_helper.enhance_vietnamese_query(query)
                print(f"Enhanced query: {enhanced_query}")

            # Step 2: Initial retrieval using overlapping chunks
            initial_results = self.chroma_db.search(
                enhanced_query,
                n_results=n_results * 2,  # Get more results for reranking
                filter_dict=filters
            )

            # Step 3: Rerank the results
            reranked_results = self.reranker.rerank(
                enhanced_query,
                initial_results,
                n_results=n_results
            )

            return reranked_results

        except Exception as e:
            print(f"Search error: {e}")
            # Fallback to basic search
            return self.chroma_db.search(query, n_results=n_results, filter_dict=filters)

    def close(self):
        """Close all database connections"""
        self.postgres_db.close()
        self.chroma_db.close()
