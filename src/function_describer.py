import ollama
import google.generativeai as genai
from typing import Dict, Any
from src.cpp_analyzer import FunctionInfo

class FunctionDescriber:
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

    def generate_function_description(self, function_info: FunctionInfo) -> str:
        prompt = f"""
        Generate a concise, semantic description of the following C++ function for use in vector search queries.
        The description should be 1-2 sentences that capture the function's purpose, algorithm, and key characteristics.
        Focus on algorithmic concepts, data structures, and computational approach rather than implementation details.

        **Function Details:**
        - **Name:** `{function_info.name}`
        - **Parameters:** `{function_info.parameters}`
        - **Return Type:** `{function_info.return_type}`
        - **Body Preview:**
        ```cpp
        {function_info.body_preview}
        ```

        **Output:** A short description (50-100 words) suitable for semantic search.
        """

        if self.provider == 'ollama':
            return self.generate_with_ollama(prompt)
        elif self.provider == 'gemini':
            return self.generate_with_gemini(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_with_ollama(self, prompt: str) -> str:
        try:
            response = ollama.chat(
                model=self.ollama_model_name,
                messages=[
                    {"role": "system", "content": "You are a C++ code analyst specializing in algorithmic descriptions."},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "num_predict": 150
                }
            )
            return response['message']['content'].strip()
        except Exception as e:
            return f"Error generating description with Ollama: {e}"

    def generate_with_gemini(self, prompt: str) -> str:
        try:
            model = genai.GenerativeModel(self.gemini_model_name)
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=150,
                temperature=0.3
            )
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text.strip()
        except Exception as e:
            return f"Error generating description with Gemini: {e}"

    def check_relevance_to_paper(self, function_description: str, paper_summary: str) -> bool:
        """
        Check if the function description is relevant to the academic paper content.
        Returns True if relevant, False otherwise.
        """
        prompt = f"""
        Determine if the following function description is relevant to the academic paper content.
        Consider algorithmic concepts, data structures, computational approaches, and technical domains.

        **Function Description:**
        {function_description}

        **Paper Summary:**
        {paper_summary}

        **Task:**
        Analyze if the function's algorithmic concepts and purpose align with the paper's content.
        Respond with only "RELEVANT" or "NOT_RELEVANT".
        """

        if self.provider == 'ollama':
            return self.check_relevance_with_ollama(prompt)
        elif self.provider == 'gemini':
            return self.check_relevance_with_gemini(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def check_relevance_with_ollama(self, prompt: str) -> bool:
        try:
            response = ollama.chat(
                model=self.ollama_model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at determining relevance between code and academic content."},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "num_predict": 20
                }
            )
            result = response['message']['content'].strip().upper()
            return result == "RELEVANT"
        except Exception as e:
            print(f"Error checking relevance with Ollama: {e}")
            return False

    def check_relevance_with_gemini(self, prompt: str) -> bool:
        try:
            model = genai.GenerativeModel(self.gemini_model_name)
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=20,
                temperature=0.1 
            )
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            result = response.text.strip().upper()
            return result == "RELEVANT"
        except Exception as e:
            print(f"Error checking relevance with Gemini: {e}")
            return False
