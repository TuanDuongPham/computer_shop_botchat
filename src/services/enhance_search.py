from src.database.postgres import PostgresDB
from src.database.chroma import ChromaDB
from src.services.reranking import RerankerService
from src.services.vietnamese_llm_helper import VietnameseLLMHelper


class EnhancedSearchService:
    def __init__(self):
        self.postgres_db = PostgresDB().connect()
        self.chroma_db = ChromaDB().connect()
        self.vi_helper = VietnameseLLMHelper()
        self.reranker = RerankerService()

    def search(self, query, language="en", n_results=5, filters=None):
        try:
            # Step 1: Enhance/translate query if in Vietnamese
            enhanced_query = query
            if language == "vi":
                enhanced_query = self.vi_helper.enhance_vietnamese_query(query)
                print(f"Enhanced query: {enhanced_query}")

            # Step 2: Initial retrieval using overlapping chunks
            initial_results = self.chroma_db.search(
                enhanced_query,
                n_results=n_results * 3,
                filter_dict=filters
            )

            # Step 3: Rerank the results
            reranked_results = self.reranker.rerank(
                enhanced_query,
                initial_results,
                n_results=n_results * 2
            )

            # Step 4: Deduplicate by product_id
            deduped_results = self._deduplicate_products(
                reranked_results, n_results)

            return deduped_results

        except Exception as e:
            print(f"Search error: {e}")
            return self.chroma_db.search(query, n_results=n_results, filter_dict=filters)

    def _deduplicate_products(self, results, n_results=5):
        if not results or 'metadatas' not in results or not results['metadatas'][0]:
            return results

        seen_products = set()
        deduped_results = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

        for i, metadata in enumerate(results['metadatas'][0]):
            if 'product_id' in metadata:
                unique_id = metadata['product_id']
            elif 'title' in metadata:
                unique_id = metadata['title']
            else:
                unique_id = results['ids'][0][i]

            if unique_id not in seen_products:
                seen_products.add(unique_id)
                deduped_results["ids"][0].append(results["ids"][0][i])
                deduped_results["documents"][0].append(
                    results["documents"][0][i])
                deduped_results["metadatas"][0].append(metadata)
                deduped_results["distances"][0].append(
                    results["distances"][0][i])

                if len(deduped_results["ids"][0]) >= n_results:
                    break

        return deduped_results

    def close(self):
        self.postgres_db.close()
        self.chroma_db.close()
