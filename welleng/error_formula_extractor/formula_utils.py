import re
from typing import Callable, Tuple

import_functions = {
    "math": ["cos", "sin", "tan", "sqrt"]
}
builtin_functions = ["abs"]

replaces = [
    ("^", "**")
]

function_keywords = [x for v in import_functions.values() for x in v]
function_keywords.extend(builtin_functions)
print(function_keywords)

arg_extractor_pattern = r"[\w\d_]+"


def convert_source_code_to_function(function_source_code: str):
    """
    The source code should only have one function or the main function
    should be the first one.
    """
    from inspect import isfunction
    temp_dict = {}

    # Ref https://www.geeksforgeeks.org/exec-in-python/
    exec(function_source_code, temp_dict)

    function = next((item for item in temp_dict.values() if isfunction(item)))
    return function


def function_builder(formula_str: str, func_name: str) -> Tuple[Callable, set, str]:

    for old, new in replaces:
        formula_str = formula_str.replace(old, new)

    keywords = re.findall(arg_extractor_pattern, formula_str)

    args = {keyword for keyword in keywords if keyword not in function_keywords}
    # removing numbers from the args
    args = {arg for arg in args if not arg.isnumeric()}

    args_str = ", ".join(args)

    imports = []
    for module, functions in import_functions.items():
        one_import = f"from {module} import {', '.join(functions)}"
        imports.append(one_import)

    imports = "\n".join(imports)

    function_str = f"""def {func_name}({args_str}):
    {imports}
    return {formula_str}"""

    func = convert_source_code_to_function(function_str)
    return func, args, function_str
