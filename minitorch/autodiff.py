from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    # TODO: Implement for Task 1.1.
    vals_plus = list(vals)
    vals_minus = list(vals)

    vals_plus[arg] += epsilon
    vals_minus[arg] -= epsilon

    central_diff = (f(*vals_plus) - f(*vals_minus)) / (2 * epsilon)

    return central_diff


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    # # Set to keep track of visited nodes
    visited_list = set()
    topo_order: List[Variable] = []

    def dfs(visit: Variable) -> None:
        # skip the variable if it has been visited is a constant
        if visit.unique_id in visited_list or visit.is_constant():
            return
            # if variable has children -> visit and run dfs on every parent
        if not visit.is_leaf():
            for parent in visit.parents:
                if not parent.is_constant():
                    dfs(parent)

        visited_list.add(visit.unique_id)
        topo_order.insert(0, visit)

    dfs(variable)
    return topo_order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    topo_sort_order = topological_sort(variable)
    derivative_list = {}
    derivative_list[variable.unique_id] = deriv
    for v in topo_sort_order:
        derivative = derivative_list[v.unique_id]
        if v.is_leaf():
            v.accumulate_derivative(derivative)
        else:
            for var, der in v.chain_rule(derivative):
                if v.is_constant():
                    continue
                derivative_list.setdefault(var.unique_id, 0.0)
                derivative_list[var.unique_id] = derivative_list[var.unique_id] + der


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
