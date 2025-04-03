from src.database.chroma import ChromaDB
from src.services.vietnamese_llm_helper import VietnameseLLMHelper
from src.services.reranking import RerankerService
from typing import Dict, List, Any, Optional


class PolicySearchService:
    def __init__(self):
        self.chroma_db = ChromaDB().connect(collection_name="policies")
        self.vi_helper = VietnameseLLMHelper()
        self.reranker = RerankerService()

    def search_policy(self,
                      query: str,
                      language: str = "vi",
                      n_results: int = 2,
                      filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
                n_results=n_results * 2,
                filter_dict=filter_dict
            )

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
            return {
                "results": self.chroma_db.search(query, n_results=n_results, filter_dict=filter_dict),
                "original_query": query,
                "enhanced_query": query,
                "error": str(e)
            }

    def format_policy_response(self, search_results: Dict[str, Any]) -> str:
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
            try:
                # Get the section path to identify the full section
                section_path = top_metadata.get('path', '')
                section_title = top_metadata.get('title', '')

                if section_title:
                    section_filter = {"title": section_title}

                    section_results = self.chroma_db.collection.query(
                        query_texts=["relevant content"],
                        n_results=10,
                        where=section_filter
                    )

                    if section_results and len(section_results['documents'][0]) > 0:
                        all_texts = []

                        for chunk_text in section_results['documents'][0]:
                            cleaned_text = chunk_text
                            cleaned_text = cleaned_text.replace("POLICY: ", "")
                            cleaned_text = cleaned_text.replace("PATH: ", "")

                            if "RELATED TERMS:" in cleaned_text:
                                cleaned_text = cleaned_text.split(
                                    "RELATED TERMS:")[0]
                            if "COMMON QUESTIONS:" in cleaned_text:
                                cleaned_text = cleaned_text.split(
                                    "COMMON QUESTIONS:")[0]

                            all_texts.append(cleaned_text.strip())

                        # Combine section chunks
                        policy_text = "\n\n".join(all_texts)
                        return response + policy_text
            except Exception as e:
                print(f"Error retrieving full section: {e}")

        policy_text = top_document

        policy_text = policy_text.replace("POLICY: ", "")
        policy_text = policy_text.replace("PATH: ", "")

        if "RELATED TERMS:" in policy_text:
            policy_text = policy_text.split("RELATED TERMS:")[0]

        if "COMMON QUESTIONS:" in policy_text:
            policy_text = policy_text.split("COMMON QUESTIONS:")[0]

        response += policy_text.strip()

        return response

    def close(self):
        self.chroma_db.close()
