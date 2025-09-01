import os
import sys
import yaml
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cpp_analyzer import CppAnalyzer
from paper_processor import PaperProcessor
from doc_generator import DocGenerator, FunctionDocGen

def load_config(config_path: str = 'config.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_documentation(docs: dict, output_path: str):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    function_docs_path = os.path.join(output_path, 'functions')
    if not os.path.exists(function_docs_path):
        os.makedirs(function_docs_path)

    for file_name, content in docs.items():
        if file_name.startswith('functions/'):
            full_path = os.path.join(output_path, file_name)
        else:
            full_path = os.path.join(output_path, file_name)

        with open(full_path, 'w') as f:
            f.write(content)
    print(f"Documentation saved to {output_path}")

def main():
    config = load_config()

    cpp_analyzer = CppAnalyzer(repo_path=config['cpp_repo_path'])
    paper_processor = PaperProcessor(
        chroma_db_path=config['chroma_db_path'],
        embedding_model=config['embedding_model']
    )
    doc_generator = DocGenerator(config)

    orchestrator = FunctionDocGen(
        cpp_analyzer,
        paper_processor,
        doc_generator,
        config
    )

    generated_docs = orchestrator.generate_complete_documentation()

    save_documentation(generated_docs, config['output_path'])


if __name__ == '__main__':

    main()
