import numpy as np
from pip._internal.utils.misc import enum

from FrozenLakeEnv import FrozenLakeEnv
from typing import List, Tuple
import heapdict


class Node:
    def __init__(self, state, prev_node=None, prev_action=None, cost=0, terminated=0, g=0):
        self.state=state
        self.prev_node = prev_node
        self.cost = cost
        self.prev_action = prev_action
        self.is_terminated = terminated
        self.g = g


class BFSAgent():
    def __init__(self) -> None:
        self.env=None
        self.expanded_nodes = 0

    def actions_cost_calculus(self, curr_node, actions, cost) -> Tuple[List[int], int]:
        if self.env.get_initial_state==curr_node.state:
            return actions, cost
        
        if curr_node.prev_action is not None:
            actions.append(curr_node.prev_action)
        if curr_node.prev_node is not None:
            return self.actions_cost_calculus(curr_node.prev_node, actions, cost+curr_node.cost)
        else:
            return actions, cost

    def solution(self, node) -> Tuple[List[int], int, float]:
        actions, cost = self.actions_cost_calculus(node, [], 0)
        actions.reverse()
        return actions, cost, self.expanded_nodes

    def failure(self) -> Tuple[List[int], int, float]:
        return [], 0, 0

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.env = env
        self.expanded_nodes=0
        initial_state=env.get_initial_state()
        start_node=Node(initial_state)
        if env.is_final_state(initial_state):
            return self.failure()
        
        open_list=[start_node]
        close_list=set()

        while open_list:
            curr_node = open_list.pop(0)
            self.expanded_nodes += 1 
            close_list.add(curr_node.state)
             
            for i in {0,1,2,3}:
                state,cost,terminated = self.env.succ(curr_node.state)[i]
                if state is None:   #is a hole
                    continue
                self.env.set_state(curr_node.state)
                child = Node(state, curr_node, i, cost)

                states_list = []
                for node in open_list:
                    states_list.append(node.state)
                if(child.state in close_list or child.state in states_list):
                    continue
                if self.env.is_final_state(child.state):
                    return self.solution(child)
                open_list.append(child)
        return [], -1, -1
 

class DFSAgent():
    def __init__(self) -> None:
        self.env = None
        self.expanded_nodes = 0

    def actions_cost_calculus(self, curr_node, actions, cost) -> Tuple[List[int], int]:
        if curr_node.state == self.env.get_initial_state:
            return actions, cost
       
        if curr_node.prev_action is not None:
            actions.append(curr_node.prev_action)
        if curr_node.prev_node is not None:
            return self.actions_cost_calculus(curr_node.prev_node, actions, cost+curr_node.cost)
        else:
            return actions, cost
        
    def solution(self, node) -> Tuple[List[int], int, float]:
        actions, cost = self.actions_cost_calculus(node, [], 0)
        actions.reverse()
        return actions, cost, self.expanded_nodes

    @staticmethod
    def failure() -> Tuple[List[int], int, float]:
        return [], 0, 0

    def recursive_dfs_g(self, open_list, close_list)-> Tuple[List[int], int, float]:
        node = open_list.pop()
        close_list.add(node.state)

        if node.is_terminated == 1:
            if self.env.is_final_state(node.state):
                return self.solution(node)
            else: 
                return None

        self.expanded_nodes += 1

        for i in {0, 1, 2, 3}:
            state, cost, terminated = self.env.succ(node.state)[i]
            child = Node(state, node, i, cost, terminated)

            if terminated == 1:
                if self.env.is_final_state(state):
                    return self.solution(child)
                else:
                    if child.state not in close_list:
                        self.expanded_nodes += 1
                    close_list.add(child.state)    
                    continue

            states_list = []
            for curr_node in open_list:
                states_list.append(curr_node.state)

            if child.state not in close_list and child.state not in states_list:
                open_list.append(child)
                result = self.recursive_dfs_g(open_list, close_list)
                if result is not None:
                    return result
            
        return None

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.env = env
        start_node = Node(env.get_initial_state())
        open_list = [start_node]
        close_list = set()
        result1 = self.recursive_dfs_g(open_list, close_list)
        if result1 is None: 
            return self.failure()
        else: 
            return result1


class UCSAgent():
    def __init__(self) -> None:
        self.env=None
        self.expanded_nodes=0

    def actions_cost_calculus(self, curr_node, actions, cost) -> Tuple[List[int], int]:
        if curr_node.state == self.env.get_initial_state:
            return actions, cost
       
        if curr_node.prev_action != None:
            actions.append(curr_node.prev_action)
        if curr_node.prev_node != None:
            return self.actions_cost_calculus(curr_node.prev_node, actions, cost+curr_node.cost)
        else:
            return actions, cost
        
    def solution(self, node) -> Tuple[List[int], int, float]:
        actions, cost = self.actions_cost_calculus(node, [], 0)
        actions.reverse()
        return actions, cost, self.expanded_nodes


    def failure() -> Tuple[List[int], int, float]:
        return [], 0, 0


    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.env=env
        open_list=heapdict.heapdict()                       
        start_node=Node(env.get_initial_state())
        open_list[start_node]=(0,0)     #in the open_list, the key is the node and the value a tuple composed of the g_value and the state
        close_list=set()
       
        while open_list:
            curr_node = open_list.popitem()[0]   #pick node with the minimum g of the queue
            close_list.add(curr_node.state)

            if self.env.is_final_state(curr_node.state):
                return self.solution(curr_node)
            
            self.expanded_nodes += 1

            for i in {0,1,2,3}:
                state, cost, terminated = self.env.succ(curr_node.state)[i]
                if state is None:    #is a hole
                    continue
                new_cost = curr_node.g + cost
                child= Node(state, curr_node, i, cost, terminated, new_cost) #the g of the child is the g of his father + the cost of the transition

                replace=False
                for node,value in open_list.items():
                    g, state=value
                    if state == child.state and g > child.g:
                        replace=True
                        break

                states_list = []
                for node in open_list:
                    states_list.append(node.state)
                    
                if child.state  not in states_list and child.state not in close_list:
                    open_list[child]=(child.g, child.state)

                elif replace:
                    del open_list[node]
                    node.g = new_cost
                    open_list[node]=(child.g, child.state)

        return self.failure()

#Greedy not using HEAPDICT
"""
class GreedyAgent():
  
    def __init__(self) -> None:
        self.env = None
        self.expanded_nodes = 0

    def heuristic(self, s) -> float:
        curr_x, curr_y = self.env.to_row_col(s)
        goals = self.env.get_goal_states()
        if not goals:
            return 0 
        
        h_s = float('inf')
        for g in goals:
            g_x, g_y = self.env.to_row_col(g)
            Manhattan_distance = abs(g_x-curr_x) + abs(g_y-curr_y)
            h_s=min(Manhattan_distance, h_s)
        return min(h_s, 100)

    def actions_cost_calculus(self, curr_node, actions, cost) -> Tuple[List[int], int]:
        if curr_node.state == self.env.get_initial_state:
            return actions, cost
       
        if curr_node.prev_action != None:
            actions.append(curr_node.prev_action)
        if curr_node.prev_node != None:
            return self.actions_cost_calculus(curr_node.prev_node, actions, cost+curr_node.cost)
        else:
            return actions, cost
        
    def solution(self, node) -> Tuple[List[int], int, float]:
        actions, cost = self.actions_cost_calculus(node, [], 0)
        actions.reverse()
        return actions, cost, self.expanded_nodes

    @staticmethod
    def failure() -> Tuple[List[int], int, float]:
        return [], 0, 0
        

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.env = env
        start_node = Node(env.get_initial_state())
        open_list_keys = [self.heuristic(0)]
        open_dic_nodes = {}
        open_dic_nodes[self.heuristic(0)] = [start_node]
        close_list= set()
        
        while open_list_keys is not None:
            #take the min f according to the sorted list of f 
            min_heuristic = open_list_keys.pop(0)
            #take one of the node in the dict with the f value min
            node = open_dic_nodes[min_heuristic].pop()
            if len(open_dic_nodes[min_heuristic]) == 0:
                del open_dic_nodes[min_heuristic]
            close_list.add(node.state)

            if node.is_terminated == 1:
                if self.env.is_final_state(node.state):
                    return self.solution(node)
                else: 
                    self.expanded_nodes += 1
                    continue  
                    
            self.expanded_nodes += 1
            
            for i in {0, 1, 2, 3}:
                state, cost, terminated = self.env.succ(node.state)[i]
                h_curr = self.heuristic(state)

                child = Node(state, node, i, cost, terminated)

                if child.state not in close_list and child not in open_dic_nodes.values():
                    if h_curr in open_dic_nodes:
                        open_dic_nodes[h_curr].append(child) 
                    else:
                        open_dic_nodes[h_curr] = [child]
                    open_list_keys.append(h_curr)
                    open_list_keys.sort()
               
        
        return self.failure()
"""

#USING HEAPDICT 
class GreedyAgent():
  
    def __init__(self) -> None:
        self.env = None
        self.expanded_nodes = 0

    def heuristic(self, s) -> float:
        curr_x, curr_y = self.env.to_row_col(s)
        goals = self.env.get_goal_states()
        if not goals:
            return 0 
        
        h_s = float('inf')
        for g in goals:
            g_x, g_y = self.env.to_row_col(g)
            Manhattan_distance = abs(g_x-curr_x) + abs(g_y-curr_y)
            h_s=min(Manhattan_distance, h_s)
        return min(h_s, 100)

    def actions_cost_calculus(self, curr_node, actions, cost) -> Tuple[List[int], int]:
        if curr_node.state == self.env.get_initial_state:
            return actions, cost
       
        if curr_node.prev_action != None:
            actions.append(curr_node.prev_action)
        if curr_node.prev_node != None:
            return self.actions_cost_calculus(curr_node.prev_node, actions, cost+curr_node.cost)
        else:
            return actions, cost
        
    def solution(self, node) -> Tuple[List[int], int, float]:
        actions, cost = self.actions_cost_calculus(node, [], 0)
        actions.reverse()
        return actions, cost, self.expanded_nodes

    @staticmethod
    def failure() -> Tuple[List[int], int, float]:
        return [], 0, 0
        

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.env = env
        start_node = Node(env.get_initial_state())
        open_list=heapdict.heapdict()                       
        open_list[start_node]=(self.heuristic(0),0)     #in the open_list, the key is the node and the value a tuple
        #composed of the heur_value and the state
        close_list=set()
        
        while open_list:
            node = open_list.popitem()[0]
            close_list.add(node.state)

            if self.env.is_final_state(node.state):
                return self.solution(node) 
                    
            self.expanded_nodes += 1
            if (node.state is None):
                continue
            
            for i in {0, 1, 2, 3}:
                state, cost, terminated = self.env.succ(node.state)[i]
                if state is None:
                    continue
                h_curr = self.heuristic(state)
                child = Node(state, node, i, cost, terminated)
                
                states_open_list = []
                for curr_node in open_list:
                    states_open_list.append(curr_node.state)

                if  child.state not in close_list and child.state not in states_open_list:
                    open_list[child]=(h_curr, child.state)
        
        return self.failure()

#Weighted A* using HEAPDICT 
class WeightedAStarAgent():
    
    def __init__(self):
        self.env = None
        self.expanded_nodes = 0 
        self.weight = 0.5

    def actions_cost_calculus(self, curr_node, actions, cost) -> Tuple[List[int], int]:
        if curr_node.state == self.env.get_initial_state:
            return actions, cost
       
        if curr_node.prev_action != None:
            actions.append(curr_node.prev_action)
        if curr_node.prev_node != None:
            return self.actions_cost_calculus(curr_node.prev_node, actions, cost+curr_node.cost)
        else:
            return actions, cost
        
    def solution(self, node) -> Tuple[List[int], int, float]:
        actions, cost = self.actions_cost_calculus(node, [], 0)
        actions.reverse()
        return actions, cost, self.expanded_nodes

    @staticmethod
    def failure() -> Tuple[List[int], int, float]:
        return [], 0, 0
    
    def heuristic(self, s) -> float:
        curr_x, curr_y = self.env.to_row_col(s)
        goals = self.env.get_goal_states()
        if not goals:
            return 0 
        
        h_s = float('inf')
        for g in goals:
            g_x, g_y = self.env.to_row_col(g)
            Manhattan_distance = abs(g_x - curr_x) + abs(g_y - curr_y)
            h_s=min(Manhattan_distance, h_s)
        return min(h_s, 100)
    
    def weighted_f(self,state, weight_h, g)-> float:
        g_s = (1-weight_h)*g
        h_s = self.heuristic(state)
        h_s_weighted = weight_h*h_s
        res = (g_s + h_s_weighted)
        return res
    
    def search(self, env: FrozenLakeEnv, h_weight) -> Tuple[List[int], int, float]:
        self.env = env
        env.reset
        self.weight = h_weight
        start_node = Node(env.get_initial_state())
        open_list=heapdict.heapdict()                       
        open_list[start_node]=(self.weighted_f(0,self.weight,0) ,0)     #in the open_list, the key is the node and the value a tuple
        #composed of the weighted f_value and the state
        close_list=set()
       
        while open_list:
            #pick node with the minimum f of the queue and keep its value (f,state)
            node, val_node = open_list.popitem() 
            tuple = (node.state, val_node[0])  
            close_list.add(tuple)

            if self.env.is_final_state(node.state):
                return self.solution(node)
            
            self.expanded_nodes += 1

            for i in {0,1,2,3}:
                state, cost, terminated = self.env.succ(node.state)[i]
                if state is None:
                    continue
                if state == node.state:
                    continue 

                new_g = node.g + cost
                child= Node(state, node, i, cost, terminated, new_g) #the g of the child is the g of his father + the cost of the transition

                weighted_f_value = self.weighted_f(state, self.weight, new_g)

                #if child is in close and has a higher f value, re-add it to open           
                replace_close = False
                for state_close, f_close in close_list:
                    if state_close == child.state and f_close > weighted_f_value:
                        replace_close=True
                        break

                #if child is in open and has a higher f value,replace
                replace_open = False
                for curr_node1,value_curr in open_list.items():
                    f, state_curr=value_curr
                    if state_curr == child.state and f > weighted_f_value:
                        replace_open=True
                        break

                open_states_list = []
                for curr_node3 in open_list:
                    open_states_list.append(curr_node3.state)

                close_states_list = []
                for state_close1, f_close1 in close_list:
                         close_states_list.append(state_close1)
                         

                if child.state  not in open_states_list and child.state not in close_states_list:
                    open_list[child]=(weighted_f_value, child.state)

                elif replace_open:
                    #remove the old node and add it back with new f value 
                    del open_list[curr_node1]
                    open_list[child]=(weighted_f_value, child.state)

                elif replace_close:
                    #remove from close list and add it to open list back with new f value 
                    tuple_to_remove = (child.state, f_close) 
                    close_list.remove(tuple_to_remove)
                    open_list[child]=(weighted_f_value, child.state)

        return self.failure()

#weighted A* not using HEAPDICT
"""
class WeightedAStarAgent():
    
    def __init__(self):
        self.env = None
        self.expanded_nodes = 0 

    def actions_cost_calculus(self, curr_node, actions, cost) -> Tuple[List[int], int]:
        if curr_node.state == self.env.get_initial_state:
            return actions, cost
       
        if curr_node.prev_action != None:
            actions.append(curr_node.prev_action)
        if curr_node.prev_node != None:
            return self.actions_cost_calculus(curr_node.prev_node, actions, cost+curr_node.cost)
        else:
            return actions, cost
        
    def solution(self, node) -> Tuple[List[int], int, float]:
        actions, cost = self.actions_cost_calculus(node, [], 0)
        actions.reverse()
        return actions, cost, self.expanded_nodes

    @staticmethod
    def failure() -> Tuple[List[int], int, float]:
        return [], 0, 0
    
    def heuristic(self, s) -> float:
        curr_x, curr_y = self.env.to_row_col(s)
        goals = self.env.get_goal_states()
        if not goals:
            return 0 
        
        h_s = float('inf')
        for g in goals:
            g_x, g_y = self.env.to_row_col(g)            
            Manhattan_distance = abs(g_x-curr_x) + abs(g_y-curr_y)
            h_s=min(Manhattan_distance, h_s)
        return min(h_s, 100)
    
    def weighted_f(self,state, weight_h, prev_node, cost_to)-> float:
        g_s = 0
        if prev_node is not None:
            g_s = prev_node.g + cost_to
        h_s = self.heuristic(state)
        return (g_s + weight_h*h_s)
        
    
    def search(self, env: FrozenLakeEnv, h_weight) -> Tuple[List[int], int, float]:
        self.env = env
        start_node = Node(env.get_initial_state())
        f_start_node = self.weighted_f(0, h_weight, None, 0)
        open_list_keys = [f_start_node]
        open_dic_nodes = {}
        open_dic_nodes[f_start_node] = [start_node]
        close_list = set()
        
        while open_list_keys is not None:
            #take the min f according to the sorted list of f 
            min_f_value = open_list_keys.pop(0)
            #take one of the node in the dic with the f value min
            node = open_dic_nodes[min_f_value].pop()
            if len(open_dic_nodes[min_f_value]) == 0:
                del open_dic_nodes[min_f_value]
            close_list.add(node.state)

            if node.is_terminated == 1:
                if self.env.is_final_state(node.state):
                    return self.solution(node)
                else: 
                    self.expanded_nodes += 1
                    continue  
                    
            self.expanded_nodes += 1

            for i in {0, 1, 2, 3}:
                state, cost, terminated = self.env.succ(node.state)[i]
                f_curr = self.weighted_f(state, h_weight, node, cost,)
                child = Node(state, node, i, cost, terminated, node.g+cost)
                #check if need to be replaced
                if child.state not in close_list and child not in open_dic_nodes.values():
                    if f_curr in open_dic_nodes:
                        open_dic_nodes[f_curr].append(child) 
                    else:
                        open_dic_nodes[f_curr] = [child]
                    open_list_keys.append(f_curr)
                    open_list_keys.sort()
            
        return self.failure()
"""


class IDAStarAgent():

    def __init__(self):
        self.env = None

    def heuristic(self, s) -> float:
        curr_x, curr_y = self.env.to_row_col(s)
        goals = self.env.get_goal_states()
        if not goals:
            return 0 
        h_s = float('inf')
        for g in goals:
            g_x, g_y = self.env.to_row_col(g)
            Manhattan_distance = abs(g_x-curr_x) + abs(g_y-curr_y)
            h_s=min(Manhattan_distance, h_s)
        return min(h_s, 100)


    def actions_calculus(self, curr_node, actions) -> List[int]:
        if curr_node.state == self.env.get_initial_state():
            return actions
       
        if curr_node.prev_action != None:
            actions.append(curr_node.prev_action)
        if curr_node.prev_node != None:
            return self.actions_calculus(curr_node.prev_node, actions)
        else:
            return actions
        
    def solution(self, node) -> Tuple[List[int], float, int]:
        actions = self.actions_calculus(node, [] )
        actions.reverse()
        return actions, node.g, -1 
    
    def failure (self)->Tuple[List[int], float, int]:
        return None, -1,-1

    def dfs_f(self, node, f_limit, path):

        new_f = node.g + self.heuristic(node.state)

        if new_f > f_limit:
            self.new_limit = min(self.new_limit, new_f)
            return self.failure()

        if self.env.is_final_state(node.state):
            return self.solution(node)

        for i in {0,1,2,3}:
            state, cost, terminated = self.env.succ(node.state)[i]
            if state is None:
                break

            child = Node(state, node, i, cost, 0, node.g + cost)

            if child.state in path:
                continue
            path.add(child.state)

            result = self.dfs_f(child, f_limit, path)
            if result[0] is not None:
                return result

            path.remove(child.state)

        return self.failure()

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.new_limit = float('inf')
        initial_state = self.env.get_initial_state()
        nodes_in_path = set()

        start_node = Node(initial_state)
        self.new_limit = self.heuristic(initial_state)

        while True:
            f_limit = self.new_limit
            self.new_limit = float('inf')

            result = self.dfs_f(start_node, f_limit, nodes_in_path)

            if result[0] is not None:
                return result

        return self.failure()

    