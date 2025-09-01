import os
import re
from typing import List, Dict
import fitz  # PyMuPDF
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

class PaperProcessor:
    def __init__(self, chroma_db_path: str, embedding_model: str):
        self.chroma_db_path = chroma_db_path
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(path=self.chroma_db_path)
        self.collection = self.client.get_or_create_collection(name="academic_paper")

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, str]:
        doc = fitz.open(pdf_path)
        sections = {}
        current_section = "Introduction"
        sections[current_section] = ""

        section_patterns = [
            r"Abstract", r"Introduction", r"Methods", r"Methodology", 
            r"Results", r"Discussion", r"Conclusion", r"References"
        ]

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            for pattern in section_patterns:
                if re.match(pattern, text, re.IGNORECASE):
                    current_section = pattern
                    if current_section not in sections:
                        sections[current_section] = ""
            
            sections[current_section] += text
        
        return sections

    def extract_algorithms_and_formulas(self, sections: Dict[str, str]) -> Dict[str, List]:
        extractions = {'algorithms': [], 'formulas': []}
        
        algo_pattern = re.compile(r"(Algorithm|Procedure|Pseudocode)\s+\d+:.*?(?=(Algorithm|Procedure|Pseudocode|$))", re.DOTALL | re.IGNORECASE)
        formula_pattern = re.compile(r"(\(Eq\.?\s*\d+\)|\[\d+\]|\(\d+\))", re.IGNORECASE)

        for _, content in sections.items():
            found_algos = algo_pattern.findall(content)
            if found_algos:
                extractions['algorithms'].extend([algo[0] for algo in found_algos])

            for line in content.split('\n'):
                if formula_pattern.search(line):
                    extractions['formulas'].append(line.strip())
        
        return extractions

    def chunk_text(self, sections: Dict[str, str], chunk_size=1000, chunk_overlap=200) -> List[Dict]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        chunks = []
        for section, content in sections.items():
            split_texts = text_splitter.split_text(content)
            for i, text in enumerate(split_texts):
                chunks.append({
                    'text': text,
                    'metadata': {'section': section, 'chunk_num': i}
                })
        return chunks

    def add_to_rag_db(self, chunks: List[Dict]):
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query_rag_db(self, query: str, n_results=5) -> List[Dict]:
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results

    def process_paper(self, pdf_path: str):
        sections = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(sections)
        self.add_to_rag_db(chunks)
        print(f"Processed paper and added {len(chunks)} chunks to the RAG database.")
