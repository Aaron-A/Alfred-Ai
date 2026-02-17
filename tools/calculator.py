"""
Calculator Tool
Evaluate mathematical expressions safely.

Uses Python's ast module to parse and evaluate math — no eval(), no exec().
Supports arithmetic, exponents, trig, log, sqrt, abs, min/max, and constants.
"""

import ast
import math
import operator
from core.tools import ToolRegistry, ToolParameter
from core.logging import get_logger

logger = get_logger("calculator")

# Safe operators
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Safe functions
_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "int": int,
    "float": float,
    # Math functions
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "log": math.log,       # natural log (or log(x, base))
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "pow": math.pow,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "radians": math.radians,
    "degrees": math.degrees,
    "hypot": math.hypot,
}

# Safe constants
_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}


def register(registry: ToolRegistry):
    """Register calculator tool."""
    registry.register_function(
        name="calculator",
        description=(
            "Evaluate a mathematical expression and return the result. "
            "Supports: +, -, *, /, //, %, ** (power), parentheses, "
            "and functions like sqrt(), sin(), cos(), log(), round(), abs(), "
            "min(), max(), factorial(). Constants: pi, e, tau. "
            "Examples: '2 ** 10', 'sqrt(144) + log(100, 10)', 'sin(pi/4)'"
        ),
        fn=calculate,
        parameters=[
            ToolParameter("expression", "string", "Math expression to evaluate, e.g. '(42 * 3.14) + sqrt(16)'"),
        ],
        category="utility",
        source="shared",
        file_path=__file__,
    )


def calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    expression = expression.strip()
    if not expression:
        return "Error: Empty expression."

    # Limit length to prevent abuse
    if len(expression) > 500:
        return "Error: Expression too long (max 500 chars)."

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        return f"Syntax error: {e}"

    try:
        result = _eval_node(tree.body)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except OverflowError:
        return "Error: Result too large."
    except ValueError as e:
        return f"Math error: {e}"
    except TypeError as e:
        return f"Type error: {e}"
    except Exception as e:
        return f"Error: {e}"

    # Format result
    if isinstance(result, float):
        # Clean up floating point noise
        if result == int(result) and not math.isinf(result):
            return str(int(result))
        # Limit decimal places for readability
        return f"{result:.10g}"
    return str(result)


def _eval_node(node):
    """Recursively evaluate an AST node. Only allows safe operations."""

    # Numbers
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    # Named constants (pi, e, tau)
    if isinstance(node, ast.Name):
        name = node.id.lower()
        if name in _CONSTANTS:
            return _CONSTANTS[name]
        raise ValueError(f"Unknown variable '{node.id}'. Available: {', '.join(_CONSTANTS.keys())}")

    # Unary operators (-x, +x)
    if isinstance(node, ast.UnaryOp):
        op = _OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))

    # Binary operators (x + y, x ** y, etc.)
    if isinstance(node, ast.BinOp):
        op = _OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        # Guard against huge exponents
        if isinstance(node.op, ast.Pow):
            if isinstance(right, (int, float)) and abs(right) > 10000:
                raise OverflowError("Exponent too large (max 10000)")
        return op(left, right)

    # Function calls: sqrt(x), log(x, 10), etc.
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are supported (e.g., sqrt(x))")

        func_name = node.func.id.lower()
        if func_name not in _FUNCTIONS:
            available = ", ".join(sorted(_FUNCTIONS.keys()))
            raise ValueError(f"Unknown function '{node.func.id}'. Available: {available}")

        args = [_eval_node(arg) for arg in node.args]
        return _FUNCTIONS[func_name](*args)

    # Lists/tuples for min/max/sum: min(1, 2, 3) or sum([1, 2, 3])
    if isinstance(node, ast.List) or isinstance(node, ast.Tuple):
        return [_eval_node(el) for el in node.elts]

    raise ValueError(f"Unsupported expression: {ast.dump(node)}")
