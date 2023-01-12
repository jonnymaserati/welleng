import re
from inspect import isfunction
from typing import Callable, Tuple

# functions under math library used in the formulas. This dict is used to manager imports (from {key} import {value})
import_functions = {
    "math": ["cos", "sin", "tan", "sqrt", "pi"]
}
# the built in python functions included in the function in EDM.
builtin_functions = ["abs", "max"]

# replace ^ with power from python.
replaces = [
    ("^", "**"),
]

function_keywords = [x for v in import_functions.values() for x in v]
function_keywords.extend(builtin_functions)

arg_extractor_pattern = r"[\w\d_]+"


def convert_source_code_to_function(function_source_code: str) -> Callable:
    """
    The source code should only have one function or the main function, a python function in string format.
    """
    temp_dict = {}

    # Ref https://www.geeksforgeeks.org/exec-in-python/
    exec(function_source_code, temp_dict)

    function = next((item for item in temp_dict.values() if isfunction(item)))
    return function


def function_builder(formula_str: str, func_name: str) -> Tuple[Callable, set, str]:
    """
    This function converts a string of formula from the EDM to the function with func_name and outputs the
    function, the arguments used and the string of the created function.
    """

    # updated the formula string to replace the keys mentioned in "replaces" list
    for old, new in replaces:
        formula_str = formula_str.replace(old, new)

    # identify the keyword in the string that follow the specified pattern.
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
