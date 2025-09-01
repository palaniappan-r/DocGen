import os
from typing import List, Dict, Any
import ollama
import google.generativeai as genai
from src.cpp_analyzer import FunctionInfo
from src.paper_processor import PaperProcessor
from src.function_describer import FunctionDescriber
import pprint

class DocGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = config.get('llm_provider', 'ollama')

        if self.provider == 'ollama':
            self.ollama_base_url = config.get('ollama_base_url', 'http://localhost:11434')
            self.ollama_model_name = config.get('ollama_model_name', 'llama3.1')
        elif self.provider == 'gemini':
            self.gemini_api_key = config.get('gemini_api_key')
            self.gemini_model_name = config.get('gemini_model_name', 'gemini-1.5-flash')
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)
            else:
                raise ValueError("Gemini API key is required when using Gemini provider")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def create_function_documentation(self, function_info: FunctionInfo, relevant_paper_content: List[Dict], config: Dict[str, Any], include_academic_context: bool = True) -> str:
        if include_academic_context and relevant_paper_content:
            context_str = "\n".join([doc['text'] for doc in relevant_paper_content])

            prompt = f"""
            Generate professional markdown documentation for the following C++ function.
            The documentation should bridge the gap between the code implementation and the concepts from the provided academic paper context.

            **Function Details:**
            - **Name:** `{function_info.name}`
            - **Parameters:** `{function_info.parameters}`
            - **Return Type:** `{function_info.return_type}`
            - **Algorithm Keywords:** `{', '.join(function_info.algorithm_keywords)}`

            **Function Body Preview:**
            ```cpp
            {function_info.body_preview}
            ```

            **Relevant Academic Paper Context:**
            ---
            {context_str}
            ---

            **Documentation Requirements:**
            1.  **Purpose:** A clear, concise description of what the function does.
            2.  **Algorithm:** Describe the algorithm used, linking it to the paper's concepts.
            3.  **Parameters:** List and explain each parameter.
            4.  **Academic Foundation:** Explicitly cite how the implementation relates to the paper.

            **Output Format (Markdown):**
            """
        else:
            prompt = f"""
            Generate professional markdown documentation for the following C++ function.
            Focus on the code implementation and algorithmic concepts.

            **Function Details:**
            - **Name:** `{function_info.name}`
            - **Parameters:** `{function_info.parameters}`
            - **Return Type:** `{function_info.return_type}`
            - **Algorithm Keywords:** `{', '.join(function_info.algorithm_keywords)}`

            **Function Body Preview:**
            ```cpp
            {function_info.body_preview}
            ```

            **Documentation Requirements:**
            1.  **Purpose:** A clear, concise description of what the function does.
            2.  **Algorithm:** Describe the algorithm used.
            3.  **Parameters:** List and explain each parameter.
            4.  **Implementation Notes:** Key implementation details and considerations.

            **Output Format (Markdown):**
            """

        if self.provider == 'ollama':
            return self.generate_with_ollama(prompt, config)
        elif self.provider == 'gemini':
            return self.generate_with_gemini(prompt, config)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_with_ollama(self, prompt: str, config: Dict[str, Any]) -> str:
        try:
            response = ollama.chat(
                model=self.ollama_model_name,
                messages=[
                    {"role": "system", "content": "You are a C++ documentation expert."},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "num_predict": config.get('max_doc_length', 2000)
                }
            )
            return response['message']['content']
        except Exception as e:
            return f"Error generating documentation with Ollama: {e}"

    def generate_with_gemini(self, prompt: str, config: Dict[str, Any]) -> str:
        try:
            model = genai.GenerativeModel(self.gemini_model_name)
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=config.get('max_doc_length', 2000),
                temperature=0.7
            )
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            return f"Error generating documentation with Gemini: {e}"

class FunctionDocGen:
    def __init__(self, cpp_analyzer, paper_processor, doc_generator, config):
        self.cpp_analyzer = cpp_analyzer
        self.paper_processor = paper_processor
        self.doc_generator = doc_generator
        self.function_describer = FunctionDescriber(config)
        self.config = config

    def generate_complete_documentation(self) -> Dict[str, str]:
        print("Analyzing C++ repository...")
        functions = self.cpp_analyzer.analyze_repository()

        print("Processing paper..")
        self.paper_processor.process_paper(self.config['paper_path'])

        paper_sections = self.paper_processor.extract_text_from_pdf(self.config['paper_path'])
        paper_summary = paper_sections.get("Abstract", "") + " " + paper_sections.get("Introduction", "")

        print("3. Generating function descriptions...")
        for i, func in enumerate(functions):
            print(f"  - Generating description for `{func.name}` ({i+1}/{len(functions)})")
            func.function_description = self.function_describer.generate_function_description(func)

        all_docs = {}


        print(f"Generating documentation for {len(functions)} functions...")
        for i, func in enumerate(functions):
            print(f"  - Processing `{func.name}` ({i+1}/{len(functions)})")

            is_relevant = self.function_describer.check_relevance_to_paper(
                func.function_description,
                paper_summary
            )

            if is_relevant:
                print(f"    - Function is relevant to paper, including academic context")
                query = func.function_description if func.function_description else f"{func.name} {' '.join(func.algorithm_keywords)}"
                relevant_content = self.paper_processor.query_rag_db(query, n_results=3)
                docs_for_func = [{'text': doc} for doc in relevant_content['documents'][0]]
                include_academic_context = True
            else:
                print(f"    - Function is not relevant to paper, generating description only")
                docs_for_func = []
                include_academic_context = False

            doc_content = self.doc_generator.create_function_documentation(
                func, docs_for_func, self.config, include_academic_context
            )
            doc_content += f"\n\n\nFile path: {func.file_path}\n"
            all_docs[f"functions/{func.name}.md"] = doc_content

        return all_docs
