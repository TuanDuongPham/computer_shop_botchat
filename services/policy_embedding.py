import re
import uuid
from typing import List, Dict, Tuple, Any, Optional
import markdown
from bs4 import BeautifulSoup


class PolicyEmbeddingService:
    def __init__(self, chroma_db, chunk_size=512, chunk_overlap=128):
        self.chroma_db = chroma_db
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.policy_collection = "policies"

        # Common policy terms and their synonyms for enhancement
        self.policy_synonyms = {
            "bảo hành": ["warranty", "guarantee", "bao hanh", "bao dam", "đảm bảo", "sửa chữa miễn phí"],
            "đổi trả": ["return", "exchange", "refund", "doi tra", "hoan tien", "hoàn tiền", "tra lai", "trả lại"],
            "thanh toán": ["payment", "pay", "purchase", "transaction", "thanh toan", "tra tien", "trả tiền"],
            "giao hàng": ["delivery", "shipping", "dispatch", "ship", "giao hang", "van chuyen", "vận chuyển"],
            "trả góp": ["installment", "credit", "loan", "financing", "tra gop", "gop", "mua trước trả sau"],
            "bảo mật": ["privacy", "security", "protection", "bao mat", "an toan", "an toàn"],
            "khiếu nại": ["complaint", "claim", "grievance", "khieu nai", "phan nan", "phàn nàn"],
            "thành viên": ["member", "membership", "account", "thanh vien", "khach hang", "khách hàng"],
            "ưu đãi": ["discount", "promotion", "offer", "deal", "uu dai", "khuyen mai", "khuyến mãi"]
        }

    def parse_policy_markdown(self, markdown_text: str) -> List[Dict[str, Any]]:
        html = markdown.markdown(markdown_text)
        soup = BeautifulSoup(html, 'html.parser')

        sections = []
        current_paths = {1: "", 2: "", 3: "", 4: "", 5: "", 6: ""}

        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'li']):
            tag_name = element.name

            if tag_name.startswith('h'):
                level = int(tag_name[1])
                title = element.get_text().strip()
                current_paths[level] = title

                for l in range(level + 1, 7):
                    current_paths[l] = ""

                path = " / ".join([p for p in list(current_paths.values())[:level] if p])

                current_section = {
                    "title": title,
                    "level": level,
                    "path": path,
                    "content": "",
                    "subsections": []
                }

                sections.append(current_section)

            else:
                if sections:
                    if tag_name == 'li':
                        sections[-1]['content'] += f"- {element.get_text().strip()}\n"
                    else:
                        sections[-1]['content'] += f"{element.get_text().strip()}\n"

        structured_sections = []
        for i, section in enumerate(sections):
            if section['level'] == 1 or section['level'] == 2:
                structured_sections.append(section)
            else:
                for j in range(i-1, -1, -1):
                    if sections[j]['level'] < section['level']:
                        sections[j]['subsections'].append(section)
                        break

        return sections

    def create_policy_chunks(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        chunks = []

        for section in sections:
            title = section['title']
            content = section['content']
            full_text = f"{title}\n\n{content}"

            # Create chunks that respect paragraph boundaries
            paragraphs = re.split(r'\n+', full_text)
            paragraphs = [p for p in paragraphs if p.strip()]

            current_chunk = []
            current_length = 0

            for paragraph in paragraphs:
                paragraph_length = len(paragraph)

                if current_length + paragraph_length > self.chunk_size and current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "title": section['title'],
                            "path": section['path'],
                            "level": section['level'],
                            "type": "policy",
                            "policy_id": str(uuid.uuid4())
                        }
                    })

                    # Start a new chunk with overlap
                    overlap_start = max(
                        0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:]
                    current_length = sum(len(p) for p in current_chunk)

                # Add the paragraph to the current chunk
                current_chunk.append(paragraph)
                current_length += paragraph_length

            # Add the last chunk if it's not empty
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "title": section['title'],
                        "path": section['path'],
                        "level": section['level'],
                        "type": "policy",
                        "policy_id": str(uuid.uuid4())
                    }
                })

            # Process subsections recursively
            if section['subsections']:
                subsection_chunks = self.create_policy_chunks(
                    section['subsections'])
                chunks.extend(subsection_chunks)

        return chunks

    def enhance_policy_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        original_text = chunk["text"]
        title = chunk["metadata"]["title"]
        path = chunk["metadata"]["path"]

        # Add title prefix
        enhanced_text = f"POLICY: {title}\nPATH: {path}\n\n{original_text}"

        # Add relevant synonyms based on content
        synonyms_to_add = []
        for term, synonyms in self.policy_synonyms.items():
            if term.lower() in original_text.lower():
                synonyms_to_add.extend(synonyms)

        if synonyms_to_add:
            synonym_text = " ".join(set(synonyms_to_add))
            enhanced_text += f"\n\nRELATED TERMS: {synonym_text}"

        # Add common question patterns based on section content
        policy_questions = self._generate_policy_questions(
            title, original_text)
        if policy_questions:
            questions_text = "\n".join(policy_questions)
            enhanced_text += f"\n\nCOMMON QUESTIONS:\n{questions_text}"

        # Update the chunk with enhanced text
        chunk["text"] = enhanced_text
        return chunk

    def _generate_policy_questions(self, title: str, content: str) -> List[str]:
        questions = []

        # Map common policy sections to question templates
        question_templates = {
            "bảo hành": [
                "Chính sách bảo hành là gì?",
                "Sản phẩm {product} được bảo hành bao lâu?",
                "Làm thế nào để yêu cầu bảo hành?",
                "Có bảo hành tại nhà không?",
                "Bảo hành có bao gồm {issue} không?"
            ],
            "đổi trả": [
                "Tôi có thể đổi trả sản phẩm không?",
                "Thời hạn đổi trả là bao lâu?",
                "Có phí đổi trả không?",
                "Cần những giấy tờ gì để đổi trả?",
                "Làm thế nào để hoàn tiền?"
            ],
            "giao hàng": [
                "Phí giao hàng là bao nhiêu?",
                "Thời gian giao hàng mất bao lâu?",
                "Có giao hàng tận nhà không?",
                "Có miễn phí giao hàng không?",
                "Tôi có thể theo dõi đơn hàng không?"
            ],
            "trả góp": [
                "Có hỗ trợ mua trả góp không?",
                "Lãi suất trả góp là bao nhiêu?",
                "Cần những giấy tờ gì để mua trả góp?",
                "Thời hạn trả góp tối đa là bao lâu?",
                "Có trả góp lãi suất 0% không?"
            ]
        }

        # Check title and content against templates
        for key, templates in question_templates.items():
            if key.lower() in title.lower() or key.lower() in content.lower():
                questions.extend(templates)
                break

        return questions[:5]

    def add_policy_to_database(self, policy_chunks: List[Dict[str, Any]]) -> None:
        for i, chunk in enumerate(policy_chunks):
            # Prepare data for ChromaDB
            policy_id = chunk["metadata"]["policy_id"]
            policy_text = chunk["text"]
            metadata = chunk["metadata"]

            # Add to ChromaDB
            self.chroma_db.collection.add(
                ids=[policy_id],
                metadatas=[metadata],
                documents=[policy_text]
            )

            if (i + 1) % 10 == 0:
                print(
                    f"Added {i + 1}/{len(policy_chunks)} policy chunks to vector database")

    def process_policy_file(self, policy_file_path: str) -> None:
        try:
            with open(policy_file_path, 'r', encoding='utf-8') as f:
                policy_content = f.read()

            sections = self.parse_policy_markdown(policy_content)
            chunks = self.create_policy_chunks(sections)
            enhanced_chunks = [self.enhance_policy_chunk(
                chunk) for chunk in chunks]
            self.add_policy_to_database(enhanced_chunks)

            return True

        except Exception as e:
            print(f"Error processing policy file: {e}")
            return False
