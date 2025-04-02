# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tools for manipulating Python code.

It implements 2 classes representing unities of code:
- Function, containing all the information we need about functions: name, args,
  body and optionally a return type and a docstring.
- Program, which contains a code preface (which could be imports, global
  variables and classes, ...) and a list of Functions.
"""
import ast
from collections.abc import Iterator, MutableSet, Sequence
import dataclasses
import io
import tokenize

from absl import logging
import astunparse


@dataclasses.dataclass
class Function:
  """A parsed Python function."""

  name: str
  args: str
  body: str
  return_type: str 
  docstring: str 

  def __str__(self) -> str:
    return_type = f' -> {self.return_type}' if self.return_type else ''

    function = f'def {self.name}({self.args}){return_type}:\n'
    if self.docstring:
      # self.docstring is already indented on every line except the first one.
      # Here, we assume the indentation is always two spaces.
      new_line = '\n' if self.body else ''
      function += f'  """{self.docstring}"""{new_line}'
    # self.body is already indented.
    function += self.body + '\n\n'
    return function

  def __setattr__(self, name: str, value: str) -> None:
    # Ensure there aren't leading & trailing new lines in `body`.
    if name == 'body':
      value = value.strip('\n')
    # Ensure there aren't leading & trailing quotes in `docstring``.
    if name == 'docstring' and value is not None:
      if '"""' in value:
        value = value.strip()
        value = value.replace('"""', '')
    super().__setattr__(name, value)


@dataclasses.dataclass(frozen=True)
class Program:
  """A parsed Python program."""

  # `preface` is everything from the beginning of the code till the first
  # function is found.
  preface: str
  functions: list[Function]

  def __str__(self) -> str:
    program = f'{self.preface}\n' if self.preface else ''
    program += '\n'.join([str(f) for f in self.functions])
    return program

  def find_function_index(self, function_name: str) -> int:
    """Returns the index of input function name."""
    function_names = [f.name for f in self.functions]
    count = function_names.count(function_name)
    if count == 0:
      raise ValueError(
          f'function {function_name} does not exist in program:\n{str(self)}'
      )
    if count > 1:
      raise ValueError(
          f'function {function_name} exists more than once in program:\n'
          f'{str(self)}'
      )
    index = function_names.index(function_name)
    return index

  def get_function(self, function_name: str) -> Function:
    index = self.find_function_index(function_name)
    return self.functions[index]


class ProgramVisitor(ast.NodeVisitor):
  """Parses code to collect all required information to produce a `Program`.

  Note that we do not store function decorators.
  """

  def __init__(self, sourcecode: str):
    self._codelines: list[str] = sourcecode.splitlines()

    self._preface: str = ''
    self._functions: list[Function] = []
    self._current_function: str | None = None

  def visit_FunctionDef(self,  # pylint: disable=invalid-name
                        node: ast.FunctionDef) -> None:
    """Collects all information about the function being parsed."""
    if node.col_offset == 0:  # We only care about first level functions.
      self._current_function = node.name
      if not self._functions:
        self._preface = '\n'.join(self._codelines[:node.lineno - 1])
      function_end_line = node.end_lineno
      body_start_line = node.body[0].lineno - 1
      # Extract the docstring.
      docstring = None
      if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value,
                                                           ast.Str):
        docstring = f'  """{ast.literal_eval(ast.unparse(node.body[0]))}"""'
        if len(node.body) > 1:
          body_start_line = node.body[1].lineno - 1
        else:
          body_start_line = function_end_line

      self._functions.append(Function(
          name=node.name,
          args=ast.unparse(node.args),
          return_type=ast.unparse(node.returns) if node.returns else None,
          docstring=docstring,
          body='\n'.join(self._codelines[body_start_line:function_end_line]),
      ))
    self.generic_visit(node)

  def return_program(self) -> Program:
    return Program(preface=self._preface, functions=self._functions)


def text_to_program(text: str) -> Program:
  """Returns Program object by parsing input text using Python AST."""
  try:
    # We assume that the program is composed of some preface (e.g. imports,
    # classes, assignments, ...) followed by a sequence of functions.
    tree = ast.parse(text)
    visitor = ProgramVisitor(text)
    visitor.visit(tree)
    return visitor.return_program()
  except Exception as e:
    logging.warning('Failed parsing %s', text)
    raise e


def text_to_function(text: str) -> Function:
  """Returns Function object by parsing input text using Python AST."""
  program = text_to_program(text)
  if len(program.functions) != 1:
    raise ValueError(f'Only one function expected, got {len(program.functions)}'
                     f':\n{program.functions}')
  return program.functions[0]


def _tokenize(code: str) -> Iterator[tokenize.TokenInfo]:
  """Transforms `code` into Python tokens."""
  code_bytes = code.encode()
  code_io = io.BytesIO(code_bytes)
  return tokenize.tokenize(code_io.readline)


def _untokenize(tokens: Sequence[tokenize.TokenInfo]) -> str:
  """Transforms a list of Python tokens into code."""
  code_bytes = tokenize.untokenize(tokens)
  return code_bytes.decode()


def _yield_token_and_is_call(
    code: str) -> Iterator[tuple[tokenize.TokenInfo, bool]]:
  """Yields each token with a bool indicating whether it is a function call."""
  try:
    tokens = _tokenize(code)
    prev_token = None
    is_attribute_access = False
    for token in tokens:
      if (prev_token and  # If the previous token exists and
          prev_token.type == tokenize.NAME and  # it is a Python identifier
          token.type == tokenize.OP and  # and the current token is a delimiter
          token.string == '('):  # and in particular it is '('.
        yield prev_token, not is_attribute_access
        is_attribute_access = False
      else:
        if prev_token:
          is_attribute_access = (
              prev_token.type == tokenize.OP and prev_token.string == '.'
          )
          yield prev_token, False
      prev_token = token
    if prev_token:
      yield prev_token, False
  except Exception as e:
    logging.warning('Failed parsing %s', code)
    raise e


def rename_function_calls(code: str, source_name: str, target_name: str) -> str:
  """Renames function calls from `source_name` to `target_name`."""
  if source_name not in code:
    return code
  modified_tokens = []
  for token, is_call in _yield_token_and_is_call(code):
    if is_call and token.string == source_name:
      # Replace the function name token
      modified_token = tokenize.TokenInfo(
          type=token.type,
          string=target_name,
          start=token.start,
          end=token.end,
          line=token.line,
      )
      modified_tokens.append(modified_token)
    else:
      modified_tokens.append(token)
  return _untokenize(modified_tokens)

def get_functions_called(code: str) -> MutableSet[str]:
  """Returns the set of all functions called in `code`."""
  return set(token.string for token, is_call in
             _yield_token_and_is_call(code) if is_call)


def yield_decorated(code: str, module: str, name: str) -> Iterator[str]:
  """Yields names of functions decorated with `@module.name` in `code`."""
  tree = ast.parse(code)
  for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
      for decorator in node.decorator_list:
        attribute = None
        if isinstance(decorator, ast.Attribute):
          attribute = decorator
        elif isinstance(decorator, ast.Call):
          attribute = decorator.func
        if (attribute is not None
            and attribute.value.id == module
            and attribute.attr == name):
          yield node.name
          
CODE_TEMPLATE = """
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

class FunSearch:
    def run(self, func): return func
    def evolve(self, func): return func
funsearch = FunSearch()

@dataclass
class ConstructionContext:
    depot: int
    candidate: int
    distance: float
    demand: int
    vehicle_load: int
    vehicle_capacity: int
    locations: np.ndarray

@dataclass
class LocalSearchContext:
    route: Tuple[int, ...]
    candidate_nodes: Tuple[int, ...]
    distance_matrix: np.ndarray
    demand: np.ndarray
    capacity: int

def solve(data: dict) -> List[List[int]]:
    #两阶段求解框架
    # 阶段1：启发式初始解构建
    initial_routes = greedy_initial_solution(data)
    
    # 阶段2：进化局部优化
    optimized_routes = local_search_optimization(data, initial_routes)
    
    return optimized_routes

# ========== 初始解构建阶段 ==========
@funsearch.evolve
def construction_heuristic(ctx: ConstructionContext) -> float:
    #进化初始解构建策略（可调整权重)
    # part1: you should change and modify this function
    return (1 * ctx.distance )

def greedy_initial_solution(data: dict) -> List[List[int]]:
    #基于进化权重的启发式初始解生成
    routes = []
    unvisited = set(range(len(data['demand']))) - {data['depot']}
    
    while unvisited and len(routes) < data['num_vehicles']:
        # part2: you should change and modify this function
        route = [data['depot']]
        current_load = 0
        
        while True:
            candidates = [n for n in unvisited 
                         if data['demand'][n] + current_load <= data['vehicle_capacity']]
            if not candidates: break
            
            # 生成候选上下文
            contexts = [
                ConstructionContext(
                    depot=data['depot'],
                    candidate=n,
                    distance=data['distance_matrix'][route[-1]][n],
                    demand=data['demand'][n],
                    vehicle_load=current_load,
                    vehicle_capacity=data['vehicle_capacity'],
                    locations=data['locations']
                ) for n in candidates
            ]
            
            # 选择最优候选节点
            scores = [construction_heuristic(ctx) for ctx in contexts]
            next_node = candidates[np.argmin(scores)]
            
            route.append(next_node)
            current_load += data['demand'][next_node]
            unvisited.remove(next_node)
        
        route.append(data['depot'])
        routes.append(route)
    
    return routes

# ========== 局部优化阶段 ==========
@funsearch.evolve
def local_search_optimization(data: dict, routes: List[List[int]]) -> List[List[int]]:
    #基于进化策略的局部搜索
    optimized = []
    for route in routes:
        if len(route) <= 2:
            optimized.append(route)
            continue

        # 生成优化上下文
        ctx = LocalSearchContext(
            route=tuple(route),
            candidate_nodes=tuple(data['demand'].nonzero()[0]),
            distance_matrix=data['distance_matrix'],
            demand=data['demand'],
            capacity=data['vehicle_capacity']
        )

        # 应用进化优化策略
        while True:
            original_cost = sum(ctx.distance_matrix[i][j] for i, j in zip(ctx.route[:-1], ctx.route[1:]))
            best_improvement = 0.0
            best_route = route.copy()

            # 2-opt邻域评估
            for i in range(1, len(ctx.route) - 2):
                for j in range(i + 1, len(ctx.route) - 1):
                    new_cost = original_cost - ctx.distance_matrix[ctx.route[i - 1]][ctx.route[i]] \
                               - ctx.distance_matrix[ctx.route[j]][ctx.route[j + 1]] \
                               + ctx.distance_matrix[ctx.route[i - 1]][ctx.route[j]] \
                               + ctx.distance_matrix[ctx.route[i]][ctx.route[j + 1]]
                    improvement = original_cost - new_cost
                    if improvement > best_improvement:
                        best_improvement = improvement
                        new_route = list(ctx.route)
                        new_route[i:j + 1] = reversed(new_route[i:j + 1])
                        best_route = new_route

            if best_improvement <= 0:
                break
            route = best_route
            ctx.route = tuple(route)

        optimized.append(route)

    return optimized

@funsearch.run
def evaluate(data):
    routes = solve(data)
    total_distance = 0.0
    distance_matrix = data['distance_matrix']
    depot = data['depot']
    demand = data['demand']
    capacity = data['vehicle_capacity']
    
    #评估解决方案并返回状态和总行驶距离
    try:
        for i, route in enumerate(routes):
            if len(route) < 2 or route[0] != depot or route[-1] != depot:
                raise ValueError(f"Invalid route {i}: {route} - must start and end at depot")

            route_distance = 0.0
            route_demand = 0
            for j in range(len(route) - 1):
                from_node = route[j]
                to_node = route[j + 1]
                route_distance += distance_matrix[from_node][to_node]
                if to_node != depot:  # 仓库需求为0
                    route_demand += demand[to_node]

            if route_demand > capacity:
                raise ValueError(f"Route {i} overloaded: {route_demand}/{capacity}")

            total_distance += route_distance

        return "success", total_distance,
    except ValueError as e:
        print(f"Evaluation failed: {e}")
        return "fail", "",[]
"""
if __name__ == "__main__":
  funcs = get_functions_called(CODE_TEMPLATE)
  print(yield_decorated(CODE_TEMPLATE,'funsearch','evolve'))
  new_code=rename_function_calls(CODE_TEMPLATE,'greedy_initial_solution','greedy_initial_solution_hello')
  print(new_code)
  # print(funcs)