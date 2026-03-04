import sympy as sp
from sympy.core.numbers import Number
from sympy.utilities.lambdify import lambdify


def replace_constants_with_parameters(expr: sp.Expr, prefix="p"):
    """Replaces numeric constants in a SymPy expression with parameter symbols, excluding common exponents like: (0, 1, -1, 2, 3, 0.5, -0.5, all Integers).

    Args:
        expr (sp.Expr): The input SymPy expression containing numeric constants.
        prefix (str, optional): The prefix string used when generating new parameter symbols. Defaults to "p".

    Returns:
        tuple: A tuple containing:
            - sympy.Expr: The new expression with constants replaced by symbols.
            - dict: A mapping dictionary of {Symbol: original_numeric_value}.
    """
    param_values = {}
    counter = 0

    def _replace(e):
        nonlocal counter
        if isinstance(e, Number) and not e.is_Integer and e not in (0, 1, -1, 2, 3, 0.5, -0.5):
            s = sp.Symbol(f"{prefix}{counter}")
            param_values[s] = e
            counter += 1
            return s
        return e

    expr_with_params = expr.replace(lambda e: isinstance(e, Number), _replace)

    return expr_with_params, param_values
