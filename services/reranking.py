from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL
import json
import traceback


class RerankerService:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def rerank(self, query, search_results, n_results=2):
        try:
            if not search_results or 'documents' not in search_results or not search_results['documents'][0]:
                print("Empty search results, skipping reranking")
                return search_results

            # Extract the documents, ids, and distances from search results
            documents = search_results['documents'][0]
            ids = search_results['ids'][0]
            distances = search_results['distances'][0]

            # Prepare the documents for reranking
            candidates = []
            for i, doc in enumerate(documents):
                metadata = search_results['metadatas'][0][i] if 'metadatas' in search_results and search_results['metadatas'][0] else {
                }
                candidates.append({
                    "id": ids[i],
                    "content": doc,
                    "score": float(distances[i]),
                    "metadata": metadata
                })

            # Prepare the prompt for reranking
            prompt = f"""
            You are a computer hardware expert. Analyze these product or policy descriptions and rerank them based on 
            how well they match the query: "{query}".
            
            Consider:
            1. Query intent and relevance to the content
            2. Specific details mentioned in the query
            3. Price or time considerations (if mentioned)
            4. Brand or specificity preferences (if mentioned)
            
            For each item, assign a score from 0-10 where 10 is perfect match.
            
            Items to evaluate:
            {json.dumps(candidates, indent=2)}
            
            Return a JSON object with a "rankings" array containing reranked IDs and scores:
            {{
            "rankings": [
                {{"id": "item_id", "score": score}},
                ...
            ]
            }}
            """

            # Call OpenAI to rerank the results
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )

            result_json = response.choices[0].message.content
            rerank_result = json.loads(result_json)
            reranked_items = None
            if isinstance(rerank_result, dict):
                for key in rerank_result:
                    if isinstance(rerank_result[key], list) and len(rerank_result[key]) > 0:
                        reranked_items = rerank_result[key]
                        break

                if reranked_items is None and "rankings" in rerank_result:
                    reranked_items = rerank_result["rankings"]
            elif isinstance(rerank_result, list):
                reranked_items = rerank_result

            if not reranked_items:
                print(f"Could not extract rankings from: {rerank_result}")
                return search_results

            # Sort by score if available
            if all('score' in item for item in reranked_items):
                reranked_items = sorted(
                    reranked_items, key=lambda x: x.get("score", 0), reverse=True)

            reranked_ids = []
            for item in reranked_items[:n_results]:
                if isinstance(item, dict) and "id" in item:
                    reranked_ids.append(item["id"])

            if not reranked_ids:
                print("No valid IDs found in reranking result")
                return search_results

            reranked_results = {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }

            # Map reranked IDs back to original content
            id_to_index = {id: i for i, id in enumerate(ids)}

            for id in reranked_ids:
                if id in id_to_index:
                    idx = id_to_index[id]
                    reranked_results["ids"][0].append(ids[idx])
                    reranked_results["documents"][0].append(documents[idx])
                    if 'metadatas' in search_results and search_results['metadatas'][0]:
                        reranked_results["metadatas"][0].append(
                            search_results['metadatas'][0][idx])
                    reranked_results["distances"][0].append(distances[idx])

            return reranked_results

        except Exception as e:
            print(f"Reranking failed: {e}")
            traceback.print_exc()

            return search_results
