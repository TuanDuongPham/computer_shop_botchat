from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL
import json


class RerankerService:
    """
    A service to rerank search results using a more sophisticated model
    than the initial retrieval.
    """

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def rerank(self, query, search_results, n_results=3):
        """
        Reranks search results using the OpenAI model to better match the query intent.

        Args:
            query: The original user query
            search_results: Initial search results from ChromaDB
            n_results: Number of results to return after reranking

        Returns:
            A list of reranked results
        """
        # Extract the documents, ids, and metadatas from search results
        documents = search_results['documents'][0]
        ids = search_results['ids'][0]
        metadatas = search_results['distances'][0]

        # Prepare the documents for reranking
        candidates = []
        for i, doc in enumerate(documents):
            candidates.append({
                "id": ids[i],
                "content": doc,
                "score": float(metadatas[i]),
                "metadata": search_results['metadatas'][0][i]
            })

        # Prepare the prompt for reranking
        prompt = f"""
        You are a computer hardware expert. Analyze these product descriptions and rerank them based on 
        how well they match the query: "{query}".
        
        Consider:
        1. Query intent and relevance to product features
        2. Technical specifications match
        3. Price considerations (if mentioned)
        4. Brand preferences (if mentioned)
        
        For each product, assign a score from 0-10 where 10 is perfect match.
        
        Products to evaluate:
        {json.dumps(candidates, indent=2)}
        
        Return ONLY a JSON array with reranked product IDs and scores:
        [
          {{"id": "product_id", "score": score}},
          ...
        ]
        """

        try:
            # Call OpenAI to rerank the results
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )

            rerank_result = json.loads(response.choices[0].message.content)

            # Extract and sort the reranked results
            reranked_ids = []
            for item in sorted(rerank_result, key=lambda x: x["score"], reverse=True)[:n_results]:
                reranked_ids.append(item["id"])

            # Rebuild the results with the new order
            reranked_results = {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }

            for id in reranked_ids:
                if id in ids:
                    idx = ids.index(id)
                    reranked_results["ids"][0].append(ids[idx])
                    reranked_results["documents"][0].append(documents[idx])
                    reranked_results["metadatas"][0].append(
                        search_results['metadatas'][0][idx])
                    reranked_results["distances"][0].append(
                        search_results['distances'][0][idx])

            return reranked_results

        except Exception as e:
            print(f"Reranking failed: {e}")
            # Return original results if reranking fails
            return search_results
