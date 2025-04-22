import os
import re
import textwrap
import vrplib
import sand_box
def read_all_instances(root_folder, ending='.vrp'):
    instances = {}
    for file_name in os.listdir(root_folder):
        if file_name.endswith(ending):
            instance = read_cvrp_data(os.path.join(root_folder, file_name))
            if instance:
                instances[file_name] = instance
                print(f'Successfully read {file_name}')
            else:
                print(f'Failed to read {file_name}')
    return instances

def read_cvrp_data(file_name, ending='.vrp'):
    if file_name.endswith(ending):
        instance = vrplib.read_instance( file_name)
        if instance:
            print(f'Successfully read {file_name}')
        else:
            print(f'Failed to read {file_name}')
    data = {}
    # basic parameters
    data["vehicle_capacity"] = instance['capacity']
    data["num_vehicles"] = int(re.search(r'k(\d+)', instance['name']).group(1))
    data["depot"] = 0
    data['locations'] = [tuple(row) for row in instance['node_coord'].tolist()]
    data["num_locations"] = len(data["locations"])
    data['demand'] = instance['demand']
    data['distance_matrix']= instance['edge_weight']
    return data


if __name__ == '__main__': 
    CODE_TEMPLATE =textwrap.dedent("""
import numpy as np
import re
from dataclasses import dataclass
from typing import Tuple, List
import vrplib
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

        return "success", total_distance, routes
    except ValueError as e:
        print(f"Evaluation failed: {e}")
        return "fail", "",[]
""")

    inputs = read_all_instances('/Users/antik/Desktop/cvrp-funsearch/cvrp-funsearch/cvrp/data/cvrp/small')
    sand_box = sand_box.Sandbox()
    sand_box.run(program=CODE_TEMPLATE,function_to_run="evaluate",inputs=inputs,test_input='A-n32-k5.vrp',timeout_seconds=500)