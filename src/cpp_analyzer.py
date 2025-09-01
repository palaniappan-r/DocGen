import os
import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class FunctionInfo:
    name: str
    file_path: str
    line_start: int
    line_end: int
    parameters: List[Dict]
    return_type: str
    docstring: Optional[str]
    body_preview: str
    includes_math: bool
    algorithm_keywords: List[str]
    function_description: Optional[str] = None
    namespace: Optional[str] = None
    class_name: Optional[str] = None
    full_qualified_name: Optional[str] = None

class CppAnalyzer:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.ALGORITHM_KEYWORDS = ['sort', 'search', 'hash', 'tree', 'graph', 'dynamic programming', 'dp']
        self.MATH_KEYWORDS = ['sqrt', 'pow', 'matrix', 'vector', 'eigen', 'sin', 'cos', 'tan']
    
    def find_cpp_files(self) -> List[str]:
        cpp_files = []
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if file.endswith(('.cpp', '.hpp', '.cxx', '.hxx', '.cc')):
                    cpp_files.append(os.path.join(root, file))
        return cpp_files
    
    def extract_functions_from_text(self, content: str, file_path: str) -> List[FunctionInfo]:
        functions = []
        
        cpp_keywords = {
            'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default',
            'break', 'continue', 'return', 'goto', 'try', 'catch', 'throw',
            'new', 'delete', 'sizeof', 'typedef', 'struct', 'class', 'enum',
            'union', 'namespace', 'using', 'public', 'private', 'protected',
            'virtual', 'static', 'const', 'volatile', 'inline', 'explicit',
            'friend', 'template', 'typename', 'auto', 'register', 'extern'
        }
        
        lines = content.split('\n')
        
        function_pattern = r'^\s*(?:(?:static|virtual|inline|explicit|friend|template\s*<[^>]*>)\s+)*' \
                          r'([^;{]*?)\s+' \
                          r'([a-zA-Z_]\w*(?:\s*<[^>]*>)?\s*::\s*[a-zA-Z_]\w*(?:\s*<[^>]*>)?(?:\s*::\s*[a-zA-Z_]\w*(?:\s*<[^>]*>)?)*)' \
                          r'\s*\(([^)]*)\)\s*(?:const)?\s*(?:noexcept)?\s*(?::[^;{]*?)?\s*\{'
        
        i = 0
        while i < len(lines):
            line = lines[i]
            match = re.match(function_pattern, line.strip())
            
            if match:
                return_type, full_qualified_name, params_str = match.groups()
                
                parts = full_qualified_name.split('::')
                func_name = parts[-1].strip()
                
                if (func_name.startswith('~') or 'operator' in func_name or
                    func_name in cpp_keywords):
                    i += 1
                    continue
                
                if len(parts) >= 3:
                    namespace = parts[0].strip()
                    class_name = parts[1].strip()
                elif len(parts) == 2:
                    namespace = None
                    class_name = parts[0].strip()
                else:
                    i += 1
                    continue
                
                parameters = []
                if params_str.strip():
                    param_parts = [p.strip() for p in params_str.split(',')]
                    for param in param_parts:
                        if param and param != 'void':
                            param_match = re.match(r'(.+?)\s+(\w+)(?:\s*=\s*[^,]*)?$', param)
                            if param_match:
                                param_type, param_name = param_match.groups()
                                parameters.append({'name': param_name, 'type': param_type.strip()})
                            else:
                                parameters.append({'name': 'param', 'type': param.strip()})
                
                brace_count = 1
                line_start = i + 1
                line_end = i + 1
                j = i + 1
                
                while j < len(lines) and brace_count > 0:
                    for char in lines[j]:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                line_end = j + 1
                                break
                    j += 1
                
                if line_end > line_start:
                    body_lines = lines[line_start:line_end]
                    body_str = '\n'.join(body_lines)
                    body_preview = body_str[:500] + '...' if len(body_str) > 500 else body_str
                else:
                    body_str = ""
                    body_preview = ""
                
                algorithm_keywords = [keyword for keyword in self.ALGORITHM_KEYWORDS if keyword in body_str.lower()]
                includes_math = any(keyword in body_str.lower() for keyword in self.MATH_KEYWORDS)
                
                function_info = FunctionInfo(
                    name=func_name,
                    file_path=file_path,
                    line_start=line_start,
                    line_end=line_end,
                    parameters=parameters,
                    return_type=return_type.strip() if return_type else "",
                    docstring=None,
                    body_preview=body_preview,
                    includes_math=includes_math,
                    algorithm_keywords=algorithm_keywords,
                    namespace=namespace,
                    class_name=class_name,
                    full_qualified_name=full_qualified_name.strip()
                )
                functions.append(function_info)
                i = line_end
            else:
                i += 1
        
        return functions
    
    def analyze_repository(self) -> List[FunctionInfo]:
        cpp_files = self.find_cpp_files()
        functions = []
        
        for i, file_path in enumerate(cpp_files):
            print(f"Analyzing file {i+1}/{len(cpp_files)}: {os.path.basename(file_path)}")
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                file_functions = self.extract_functions_from_text(content, file_path)
                functions.extend(file_functions)
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
                continue
        
        print(f"Analysis complete. Total functions found: {len(functions)}")
        return functions
    
    def save_analysis(self, functions: List[FunctionInfo], output_path: str):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        output_file = os.path.join(output_path, 'cpp_analysis.json')
        
        functions_dict = [func.__dict__ for func in functions]
        with open(output_file, 'w') as f:
            json.dump(functions_dict, f, indent=4)
