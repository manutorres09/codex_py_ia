"""Simulación de búsqueda exhaustiva para reubicar un robot de montaje.

Este módulo provee utilidades para modelar el entorno lineal de la cinta
transportadora y ejecutar algoritmos de búsqueda exhaustiva (sin
información heurística) tales como la búsqueda en anchura (BFS) y la
búsqueda en profundidad (DFS). La problemática se inspira en un robot que
debe corregir una desalineación horizontal: partiendo de una posición B
desconocida debe reencontrar la posición de montaje correcta A explorando
indistintamente hacia la izquierda y hacia la derecha.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class SearchNode:
    """Representa un nodo en el árbol de búsqueda.

    Attributes
    ----------
    state: int
        Posición actual sobre la cinta transportadora.
    parent: Optional["SearchNode"]
        Nodo padre desde el cual se alcanzó el estado actual. Se utiliza
        para reconstruir la ruta de solución una vez encontrado el objetivo.
    action: Optional[str]
        Identificador simbólico de la acción ejecutada ("LEFT" o "RIGHT").
    depth: int
        Profundidad del nodo dentro del árbol de búsqueda.
    """

    state: int
    parent: Optional["SearchNode"]
    action: Optional[str]
    depth: int


class ConveyorEnvironment:
    """Entorno unidimensional que modela la cinta transportadora.

    Parameters
    ----------
    target_position: int
        Posición correcta de montaje (objetivo A).
    min_position: int
        Límite inferior de exploración permitido para el robot.
    max_position: int
        Límite superior de exploración permitido para el robot.
    step: int, default 1
        Magnitud de desplazamiento horizontal que puede realizar el robot
        en cada acción. Un valor de 1 representa un movimiento casillero a
        casillero.
    """

    def __init__(
        self,
        target_position: int,
        min_position: int,
        max_position: int,
        *,
        step: int = 1,
    ) -> None:
        if min_position > max_position:
            raise ValueError("El límite inferior no puede superar al superior.")
        if not (min_position <= target_position <= max_position):
            raise ValueError("El objetivo debe encontrarse dentro de los límites.")
        if step <= 0:
            raise ValueError("El paso debe ser un entero positivo.")

        self.target_position = target_position
        self.min_position = min_position
        self.max_position = max_position
        self.step = step

    def is_goal(self, position: int) -> bool:
        """Indica si la posición actual coincide con la posición objetivo."""

        return position == self.target_position

    def successors(self, position: int) -> Iterable[Tuple[str, int]]:
        """Genera las posiciones alcanzables desde ``position``.

        El robot puede moverse un paso a la izquierda o a la derecha mientras
        permanezca dentro de los límites de seguridad definidos por la planta.
        Se devuelven pares (acción, nuevo_estado).
        """

        moves: Dict[str, int] = {
            "LEFT": position - self.step,
            "RIGHT": position + self.step,
        }

        for action, new_position in moves.items():
            if self.min_position <= new_position <= self.max_position:
                yield action, new_position


@dataclass
class SearchResult:
    """Resultado estructurado de un algoritmo de búsqueda."""

    goal_node: Optional[SearchNode]
    explored_order: List[int]

    @property
    def solution_path(self) -> List[int]:
        """Reconstruye la secuencia de estados desde el inicio hasta el objetivo."""

        if self.goal_node is None:
            return []

        path: List[int] = []
        current: Optional[SearchNode] = self.goal_node
        while current is not None:
            path.append(current.state)
            current = current.parent
        return list(reversed(path))


def breadth_first_search(env: ConveyorEnvironment, start: int) -> SearchResult:
    """Implementa la búsqueda en anchura (BFS) sobre la cinta transportadora.

    El algoritmo explora los estados por capas de profundidad creciente,
    garantizando encontrar la solución de menor número de pasos cuando el
    costo de cada desplazamiento es uniforme.
    """

    frontier: Deque[SearchNode] = deque(
        [SearchNode(state=start, parent=None, action=None, depth=0)]
    )
    explored: set[int] = set()
    explored_order: List[int] = []

    frontier_states = {start}

    while frontier:
        node = frontier.popleft()
        frontier_states.remove(node.state)
        explored_order.append(node.state)

        if env.is_goal(node.state):
            return SearchResult(goal_node=node, explored_order=explored_order)

        explored.add(node.state)

        for action, successor in env.successors(node.state):
            if successor in explored or successor in frontier_states:
                continue
            child = SearchNode(
                state=successor,
                parent=node,
                action=action,
                depth=node.depth + 1,
            )
            frontier.append(child)
            frontier_states.add(successor)

    return SearchResult(goal_node=None, explored_order=explored_order)


def depth_first_search(env: ConveyorEnvironment, start: int) -> SearchResult:
    """Implementa la búsqueda en profundidad (DFS) con detección de repetidos.

    Se emplea una pila explícita para priorizar la expansión de los nodos más
    profundos, simulando el comportamiento de un robot que explora un camino a
    fondo antes de retroceder. La detección de estados repetidos evita ciclos
    infinitos dentro del espacio acotado de la cinta.
    """

    stack: List[SearchNode] = [SearchNode(state=start, parent=None, action=None, depth=0)]
    explored: set[int] = set()
    explored_order: List[int] = []

    while stack:
        node = stack.pop()
        explored_order.append(node.state)

        if env.is_goal(node.state):
            return SearchResult(goal_node=node, explored_order=explored_order)

        if node.state in explored:
            # Si el estado ya fue desarrollado desde un camino alternativo,
            # se omite para evitar ciclos.
            continue

        explored.add(node.state)

        # Para emular una pila LIFO tradicional, los sucesores se insertan en
        # orden inverso: primero derecha, luego izquierda. Así, el primer
        # movimiento que se explorará al desempilar será a la izquierda,
        # siguiendo la intuición de "probar" un lado completo antes de cambiar.
        successors = list(env.successors(node.state))
        for action, successor in reversed(successors):
            if successor in explored:
                continue
            child = SearchNode(
                state=successor,
                parent=node,
                action=action,
                depth=node.depth + 1,
            )
            stack.append(child)

    return SearchResult(goal_node=None, explored_order=explored_order)


def _simulate_example() -> None:
    """Ejecuta una simulación demostrativa cuando el módulo se ejecuta directo."""

    env = ConveyorEnvironment(target_position=2, min_position=-5, max_position=5)
    start_position = -1

    bfs_result = breadth_first_search(env, start=start_position)
    dfs_result = depth_first_search(env, start=start_position)

    print("BÚSQUEDA EN ANCHURA (BFS)")
    print("Orden de exploración:", bfs_result.explored_order)
    print("Ruta de solución:", bfs_result.solution_path)

    print("\nBÚSQUEDA EN PROFUNDIDAD (DFS)")
    print("Orden de exploración:", dfs_result.explored_order)
    print("Ruta de solución:", dfs_result.solution_path)


if __name__ == "__main__":
    _simulate_example()
