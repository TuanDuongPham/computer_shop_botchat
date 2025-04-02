from database.chroma import ChromaDB
from services.vietnamese_llm_helper import VietnameseLLMHelper
from services.reranking import RerankerService
from typing import Dict, List, Any, Optional


class PolicySearchService:
    """
    Service for searching and retrieving policy information from the vector database.
    """

    def __init__(self):
        """Initialize the policy search service."""
        self.chroma_db = ChromaDB().connect(collection_name="policies")
        self.vi_helper = VietnameseLLMHelper()
        self.reranker = RerankerService()

    def search_policy(self,
                      query: str,
                      language: str = "vi",
                      n_results: int = 3,
                      filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search for policy information based on the query.

        Args:
            query: The search query
            language: Query language ("vi" for Vietnamese, "en" for English)
            n_results: Number of results to return
            filter_dict: Optional filter to apply to the search

        Returns:
            Dictionary with search results and metadata
        """
        try:
            # Set type filter to only search policy content
            policy_filter = {"type": "policy"}
            if filter_dict:
                # Combine with provided filters
                filter_dict = {**filter_dict, **policy_filter}
            else:
                filter_dict = policy_filter

            # Enhance query if in Vietnamese
            enhanced_query = query
            if language == "vi":
                enhanced_query = self.vi_helper.enhance_vietnamese_query(query)
                print(f"Enhanced policy query: {enhanced_query}")

            # Perform initial search
            initial_results = self.chroma_db.search(
                enhanced_query,
                n_results=n_results * 2,  # Get more for reranking
                filter_dict=filter_dict
            )

            # Rerank results for better relevance
            reranked_results = self.reranker.rerank(
                enhanced_query,
                initial_results,
                n_results=n_results
            )

            return {
                "results": reranked_results,
                "original_query": query,
                "enhanced_query": enhanced_query
            }

        except Exception as e:
            print(f"Policy search error: {e}")
            # Fallback to basic search
            return {
                "results": self.chroma_db.search(query, n_results=n_results, filter_dict=filter_dict),
                "original_query": query,
                "enhanced_query": query,
                "error": str(e)
            }

    def format_policy_response(self, search_results: Dict[str, Any]) -> str:
        """
        Format policy search results into a user-friendly response.

        Args:
            search_results: Results from search_policy method

        Returns:
            Formatted response string
        """
        if not search_results.get("results") or not search_results["results"].get("documents"):
            return "Xin lỗi, tôi không tìm thấy thông tin chính sách liên quan đến câu hỏi của bạn."

        documents = search_results["results"]["documents"][0]
        metadatas = search_results["results"]["metadatas"][0]

        # Extract the most relevant result
        top_document = documents[0]
        top_metadata = metadatas[0]

        # Format the response
        response = ""

        # Add section information
        if "title" in top_metadata:
            response += f"### {top_metadata['title']}\n\n"

        # Clean up the document text
        policy_text = top_document

        # Remove technical tags
        policy_text = policy_text.replace("POLICY: ", "")
        policy_text = policy_text.replace("PATH: ", "")

        # Remove the RELATED TERMS section
        if "RELATED TERMS:" in policy_text:
            policy_text = policy_text.split("RELATED TERMS:")[0]

        # Remove the COMMON QUESTIONS section
        if "COMMON QUESTIONS:" in policy_text:
            policy_text = policy_text.split("COMMON QUESTIONS:")[0]

        # Add the cleaned policy text
        response += policy_text.strip()

        return response

    def close(self):
        """Close the database connections."""
        self.chroma_db.close()
