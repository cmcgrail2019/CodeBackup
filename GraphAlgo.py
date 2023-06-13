"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

import heapq
import os
import pickle
import math




class PriorityQueue(object):
    counter = 0
    
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []
        
    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """

        return heapq.heappop(self.queue)


    def remove(self, node):
        """
        Remove a node from the queue.

        Hint: You might require this in ucs. However, you may
        choose not to use it or to define your own method.

        Args:
            node (tuple): The node to remove from the queue.
        """
        
        

        raise NotImplementedError

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    
    def append(self, node):
         
        
        if self.queue is []:
            node = (node[0], 0, node[1],)
        else:
            self.counter = self.counter +1
            node = (node[0], self.counter, node[1])
        
        
       
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        
        return heapq.heappush(self.queue, node)
            

        raise NotImplementedError
        
    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in the queue.
        """

        return self.queue[0]


def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    frontier = []
    visited = []
    solutionset = []
    parentset = {}
    parentset[start] = None
    costset = {}
    costset[start] = 0

    if start == goal:
        return solutionset

    #Append first item to the frontier
    frontier.append((0,start)) 
    visited.append(start)


    while frontier:

        node = heapq.heappop(frontier)
        #print(node)

        #node = frontier.pop()
        #print(node)
        nodekey = node[1]
        #print(nodekey)
        children = graph[nodekey].copy()
        sortlist = []
        for neighbor in children: 
            sortlist.append(neighbor)
            if neighbor not in costset:
                costset[neighbor] = float("inf")
        sorted(sortlist)



            #If solution is in child nodes, evaluate child node immediately 
        #if goal in children:
        #    print('in goal loop')
        #    visited.append(goal)
        #    solutionset.append(goal)
        #    parentset[goal] = nodekey
        #    while parentset[goal] is not None:
        #        solutionset.append(parentset[goal]) 
        #        goal = parentset[goal]
        #    solutionset.reverse()
            #print(solutionset)

            #print(visited)

            #otherwise, add child nodes to the frontier
        for neighbor in sorted(sortlist):
            if neighbor not in visited:
                visited.append(neighbor)
                if costset[neighbor] > (costset[nodekey] + 1):                    
                    costset[neighbor] = (costset[nodekey] + 1)
                    #print('Frontier', (costset[nodekey],(neighbor, graph[neighbor])))
                    frontier.append((costset[nodekey],neighbor))
                    parentset[neighbor] = nodekey
                    if neighbor == goal:
                        solutionset.append(neighbor)
                        while parentset[goal] is not None:
                            solutionset.append(parentset[goal]) 
                            goal = parentset[goal]
                        solutionset.reverse()
                        return solutionset

                


    if node in visited:
        heapq.heappop(frontier)
        node = frontier[0]

    raise NotImplementedError



def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    #graph.get_edge_weight(node_1, node_2)

    
    frontier = PriorityQueue()
    visited = []
    solutionset = []
    parentset = {}
    costset = {}
    parentset[start] = None  
    costset[start] = 0

    #if start == goal
    if start == goal:
        return solutionset


    #Append first item to the frontier
    frontier.append((0,start))

    while frontier:
        nodeset = frontier.pop()
        nodekey = nodeset[2]

        if nodekey == goal:
            solutionset.append(nodekey)
            while parentset[goal] is not None:
                solutionset.append(parentset[goal]) 
                goal = parentset[goal]
            solutionset.reverse()
            return solutionset
        children = graph[nodekey].copy()

        for c in children:
            if c not in costset:
                costset[c] = float("inf") 


        #update frontier to include child nodes
        for neighbor in children:
            if neighbor not in visited:
                visited.append(nodekey)
                if costset[neighbor] > (costset[nodekey] + graph.get_edge_weight(nodekey, neighbor)):                    
                    costset[neighbor] = (costset[nodekey] + graph.get_edge_weight(nodekey, neighbor))
                    frontier.append((costset[neighbor],neighbor)) 
                    parentset[neighbor] = nodekey
    if frontier.size() == 0:
        return 'No Solution'




    raise NotImplementedError


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """
    goalnode = graph.nodes[goal]['pos']
    node = graph.nodes[v]['pos']
    
    
    x = math.pow(goalnode[0] - node[0], 2.0);
    y = math.pow(goalnode[1] - node[1], 2.0);

    return math.sqrt(x + y)
        



    # TODO: finish this function!
    raise NotImplementedError


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """


    frontier = PriorityQueue()
    visited = []
    solutionset = []
    parentset = {}
    costset = {}
    parentset[start] = None  
    costset[start] = 0

    #if start == goal
    if start == goal:
        return solutionset


    #Append first item to the frontier
    frontier.append((0,start))

    while frontier:
        nodeset = frontier.pop()
        nodekey = nodeset[2]

        # new cost = cost of parent(Actual)+distance from parent to node
        if nodekey != start:
            costset[nodekey] = costset[parentset[nodekey]] + graph.get_edge_weight(parentset[nodekey], nodekey)


        if nodekey == goal:
            solutionset.append(nodekey)
            while parentset[goal] is not None:
                solutionset.append(parentset[goal]) 
                goal = parentset[goal]
            solutionset.reverse()
            return solutionset
            #print(solutionset)
            #print(costset[nodekey])
        children = graph[nodekey].copy()

        for c in children:
            if c not in costset:
                costset[c] = float("inf") 


        #update frontier to include child nodes
        for neighbor in children:
            if neighbor not in visited:
                visited.append(nodekey)
                hdis = euclidean_dist_heuristic(graph, neighbor, goal)
                if costset[neighbor] > (costset[nodekey] + graph.get_edge_weight(nodekey, neighbor) + hdis):                   
                    costset[neighbor] = (costset[nodekey] + graph.get_edge_weight(nodekey, neighbor) + hdis)
                    frontier.append((costset[neighbor],neighbor)) 
                    parentset[neighbor] = nodekey
    if frontier.size() == 0:
        return 'No Solution'

    
    raise NotImplementedError


def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    
    start_frontier = PriorityQueue()
    goal_frontier = PriorityQueue()
    start_visited = []
    goal_visited = []
    solutionset = []
    start_parentset = {}
    goal_parentset = {}
    start_costset = {}
    goal_costset = {}
    start_parentset[start] = None 
    goal_parentset[goal] = None  
    start_costset[start] = 0
    goal_costset[goal] = 0
    mu = float('inf')

    #if start == goal
    if start == goal:
        return solutionset


    #Append first item to the frontier
    start_frontier.append((0,start))
    goal_frontier.append((0,goal))

    while start_frontier or goal_frontier:

        #popping start frontier & checking if value is on goalset
        start_nodeset = start_frontier.pop()
        start_nodekey = start_nodeset[2]

        if start_nodekey not in start_visited:
            start_visited.append(start_nodekey)    

        if start_nodekey in (goal_parentset.keys()):
            intersection = start_nodekey


            if start_costset[intersection] + goal_costset[intersection] < mu:
                mu = start_costset[intersection] + goal_costset[intersection]     


            if start_frontier.size() > 0:
                start_nodecheck = start_frontier.top()
            if start_frontier.size() == 0:
                start_nodecheck = start_nodeset

            if goal_frontier.size() > 0:
                goal_nodecheck = goal_frontier.top()
            if goal_frontier.size() == 0:
                goal_nodecheck = goal_nodeset

            if mu < (start_costset[start_nodecheck[2]] + goal_costset[goal_nodecheck[2]]):

                start_visited.append(start_nodekey)
                goal_visited.append(goal_nodekey)


                    #find crossover node list
                crosscost = PriorityQueue()
                crossover = set(start_parentset.keys()).intersection(set(goal_parentset.keys()))
                for n in crossover:
                    crosscost.append((start_costset[n] + goal_costset[n],n))
                topnode = crosscost.top()[2]

                solutionset.append(topnode)
                while start_parentset[topnode] is not None:
                    solutionset.append(start_parentset[topnode]) 
                    topnode = start_parentset[topnode]
                solutionset.reverse()

                topnode = crosscost.top()[2]
                while goal_parentset[topnode] is not None:
                    solutionset.append(goal_parentset[topnode])
                    topnode = goal_parentset[topnode] 
                return solutionset


        #popping goal frontier & checking if in startset
        goal_nodeset = goal_frontier.pop()
        goal_nodekey = goal_nodeset[2]

        if goal_nodekey not in goal_visited:
            goal_visited.append(goal_nodekey)



        if goal_nodekey in (start_parentset.keys()):

            intersection = goal_nodekey


            if start_costset[intersection] + goal_costset[intersection] < mu:
                mu = start_costset[intersection] + goal_costset[intersection]     



            if start_frontier.size() > 0:
                start_nodecheck = start_frontier.top()
            if start_frontier.size() == 0:
                start_nodecheck = start_nodeset

            if goal_frontier.size() > 0:
                goal_nodecheck = goal_frontier.top()
            if goal_frontier.size() == 0:
                goal_nodecheck = goal_nodeset



            if mu < (start_costset[start_nodecheck[2]] + goal_costset[goal_nodecheck[2]]):

                start_visited.append(start_nodekey)
                goal_visited.append(goal_nodekey)


                    #find crossover node list
                crosscost = PriorityQueue()
                crossover = set(goal_parentset.keys()).intersection(set(start_parentset.keys()))
                for n in crossover:
                    crosscost.append((start_costset[n] + goal_costset[n],n))
                topnode = crosscost.top()[2]

                solutionset.append(topnode)
                while start_parentset[topnode] is not None:
                    solutionset.append(start_parentset[topnode]) 
                    topnode = start_parentset[topnode]
                solutionset.reverse()

                topnode = crosscost.top()[2]
                while goal_parentset[topnode] is not None:
                    solutionset.append(goal_parentset[topnode])
                    topnode = goal_parentset[topnode]
                return solutionset
                
                ##Goal Check Code Went Here


        start_children = graph[start_nodekey].copy()

        for c in start_children:
            if c not in start_costset:
                start_costset[c] = float("inf") 

        goal_children = graph[goal_nodekey].copy()
        for c in goal_children:
            if c not in goal_costset:
                goal_costset[c] = float("inf") 


        #update start frontier to include child nodes
        for start_neighbor in start_children:
            if start_neighbor not in start_visited:

                if start_costset[start_neighbor] > (start_costset[start_nodekey] + graph.get_edge_weight(start_nodekey, start_neighbor)):                    
                    start_costset[start_neighbor] = (start_costset[start_nodekey] + graph.get_edge_weight(start_nodekey, start_neighbor))
                    start_frontier.append((start_costset[start_neighbor],start_neighbor)) 
                    start_parentset[start_neighbor] = start_nodekey

        #update goal frontier to include child nodes
        for goal_neighbor in goal_children:
            if goal_neighbor not in goal_visited:                
                if goal_costset[goal_neighbor] > (goal_costset[goal_nodekey] + graph.get_edge_weight(goal_neighbor, goal_nodekey)):                    
                    goal_costset[goal_neighbor] = (goal_costset[goal_nodekey] + graph.get_edge_weight(goal_neighbor, goal_nodekey))
                    goal_frontier.append((goal_costset[goal_neighbor],goal_neighbor)) 
                    goal_parentset[goal_neighbor] = goal_nodekey


    if start_frontier.size() + goal_frontier.size() == 0:
        return 'No Solution'
    
    
    
    raise NotImplementedError


def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    start_frontier = PriorityQueue()
    goal_frontier = PriorityQueue()
    start_visited = []
    goal_visited = []
    solutionset = []
    start_parentset = {}
    goal_parentset = {}
    start_costset = {}
    goal_costset = {}
    start_parentset[start] = 0 
    goal_parentset[goal] = 0  
    start_costset[start] = 0
    goal_costset[goal] = 0
    mu = float('inf')

    #if start == goal
    if start == goal:
        return solutionset


    #Append first item to the frontier
    start_frontier.append((0,start))
    goal_frontier.append((0,goal))

    while start_frontier or goal_frontier:

        #popping start frontier & checking if value is on goalset
        start_nodeset = start_frontier.pop()
        start_nodekey = start_nodeset[2]

        if start_nodekey != start:
            start_costset[start_nodekey] = start_costset[start_parentset[start_nodekey]] + graph.get_edge_weight(start_parentset[start_nodekey], start_nodekey)

        if start_nodekey not in start_visited:
            start_visited.append(start_nodekey) 

        start_children = graph[start_nodekey].copy()
        
        if ((start_nodekey == start) and  (goal in start_children)):
            solutionset.append(start)
            solutionset.append(goal)
            return solutionset
        
        
        for c in start_children:
            if c not in start_costset:
                start_costset[c] = float("inf") 

            #update start frontier to include child nodes
        for start_neighbor in start_children:
            if start_neighbor not in start_visited:
                start_hdis = euclidean_dist_heuristic(graph, start_neighbor, goal)
                if start_costset[start_neighbor] > (start_costset[start_nodekey] + graph.get_edge_weight(start_nodekey, start_neighbor) + start_hdis):                    
                    start_costset[start_neighbor] = (start_costset[start_nodekey] + graph.get_edge_weight(start_nodekey, start_neighbor) + start_hdis)
                    start_frontier.append((start_costset[start_neighbor],start_neighbor)) 
                    start_parentset[start_neighbor] = start_nodekey



        if start_nodekey in goal_visited:
            intersection = start_nodekey


            if goal_parentset[intersection] != 0:
                goal_costset[intersection] = goal_costset[goal_parentset[intersection]] + graph.get_edge_weight(goal_parentset[intersection], intersection)

            if start_costset[intersection] + goal_costset[intersection] < mu:
                mu = start_costset[intersection] + goal_costset[intersection]


            if start_frontier.size() > 0:
                start_next = start_frontier.top()
                start_key = start_next[2]

            if start_frontier.size() == 0:
                start_next = start_nodeset
                start_key = start_next[2]



            if goal_frontier.size() > 0:
                goal_next = goal_frontier.top()
                goal_key = goal_next[2]


            if goal_frontier.size() == 0:
                goal_next = goal_nodeset
                goal_key = goal_next[2]


            if mu + euclidean_dist_heuristic(graph, start, goal) < (start_next[0] + goal_next[0]):

                start_visited.append(start_nodekey)
                goal_visited.append(goal_nodekey)


                    #find crossover node list
                crosscost = PriorityQueue()
                crossover = set(start_parentset.keys()).intersection(set(goal_parentset.keys()))
                
                
                
                for n in crossover:
                    if n != start and n != goal:
                        crosscost.append((start_costset[start_parentset[n]] + graph.get_edge_weight(start_parentset[n], n) + goal_costset[goal_parentset[n]] + graph.get_edge_weight(goal_parentset[n], n),n))
                topnode = crosscost.top()[2]

                solutionset.append(topnode)
                while start_parentset[topnode] != 0 :
                    solutionset.append(start_parentset[topnode]) 
                    topnode = start_parentset[topnode]
                solutionset.reverse()

                topnode = crosscost.top()[2]
                while goal_parentset[topnode] != 0 :
                    solutionset.append(goal_parentset[topnode])
                    topnode = goal_parentset[topnode]
                return solutionset



        #popping goal frontier & checking if in startset
        goal_nodeset = goal_frontier.pop()
        goal_nodekey = goal_nodeset[2]

        if goal_nodekey != goal:
            goal_costset[goal_nodekey] = goal_costset[goal_parentset[goal_nodekey]] + graph.get_edge_weight(goal_parentset[goal_nodekey], goal_nodekey)

        if goal_nodekey not in goal_visited:
            goal_visited.append(goal_nodekey)






        goal_children = graph[goal_nodekey].copy()
        for c in goal_children:
            if c not in goal_costset:
                goal_costset[c] = float("inf") 



        #update goal frontier to include child nodes
        for goal_neighbor in goal_children:
            if goal_neighbor not in goal_visited:  
                goal_hdis = euclidean_dist_heuristic(graph, goal_neighbor, start)
                if goal_costset[goal_neighbor] > (goal_costset[goal_nodekey] + graph.get_edge_weight(goal_neighbor, goal_nodekey) + goal_hdis):                    
                    goal_costset[goal_neighbor] = (goal_costset[goal_nodekey] + graph.get_edge_weight(goal_neighbor, goal_nodekey) + goal_hdis)
                    goal_frontier.append((goal_costset[goal_neighbor],goal_neighbor)) 
                    goal_parentset[goal_neighbor] = goal_nodekey


        if goal_nodekey in start_visited:

            intersection = goal_nodekey        

            if start_parentset[intersection] != 0:
                start_costset[intersection] = start_costset[start_parentset[intersection]] + graph.get_edge_weight(start_parentset[intersection], intersection)


            if start_costset[intersection] + goal_costset[intersection] < mu:
                mu = start_costset[intersection] + goal_costset[intersection]     

            if start_frontier.size() > 0:
                start_nextg = start_frontier.top()
                start_keyg = start_nextg[2]

            if start_frontier.size() == 0:
                start_nextg = start_nodeset
                start_keyg = start_nextg[2]



            if goal_frontier.size() > 0:
                goal_nextg = goal_frontier.top()
                goal_keyg = goal_nextg[2]

            if goal_frontier.size() == 0:
                goal_nextg = goal_nodeset
                goal_keyg = goal_nextg[2]


            if mu + euclidean_dist_heuristic(graph, start, goal) < (start_nextg[0] + goal_nextg[0]):

                start_visited.append(start_keyg)
                goal_visited.append(goal_keyg)



                    #find crossover node list
                crosscost = PriorityQueue()
                crossover = set(goal_parentset.keys()).intersection(set(start_parentset.keys()))
                
                
                for n in crossover:
                    if n != start and n != goal:
                        crosscost.append((start_costset[start_parentset[n]] + graph.get_edge_weight(start_parentset[n], n) + goal_costset[goal_parentset[n]] + graph.get_edge_weight(goal_parentset[n], n),n))
                topnode = crosscost.top()[2]

                solutionset.append(topnode)
                while start_parentset[topnode] != 0 :
                    solutionset.append(start_parentset[topnode]) 
                    topnode = start_parentset[topnode]
                solutionset.reverse()

                topnode = crosscost.top()[2]
                while goal_parentset[topnode] != 0 :
                    solutionset.append(goal_parentset[topnode])
                    topnode = goal_parentset[topnode]
                return solutionset


    raise NotImplementedError


def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    goalA = goals[0]
    goalB = goals[1]
    goalC = goals[2]


    first_frontier = PriorityQueue() #Searches for Goal A
    second_frontier = PriorityQueue() #Searches for Goal B
    third_frontier = PriorityQueue() #Searches for Goal C

    first_explored = []
    second_explored = []
    third_explored = []

    first_parentset = {}
    first_parentset[goalA] = 0
    second_parentset = {}
    second_parentset[goalB] = 0
    third_parentset = {}
    third_parentset[goalC] = 0

    first_costset = {}
    first_costset[goalA] = 0
    second_costset = {}
    second_costset[goalB] = 0
    third_costset = {}
    third_costset[goalC] = 0

    mu_oneB = float('inf')
    mu_oneC = float('inf')
    mu_twoA = float('inf')
    mu_twoC = float('inf')
    mu_threeA = float('inf')
    mu_threeB = float('inf')

    first_AB = False
    first_CA = False
    second_AB = False
    second_BC = False
    third_BC = False
    third_CA = False



    goalAB_found = False
    goalBC_found = False
    goalCA_found = False

    goalpathAB = []
    goalpathBC = []
    goalpathCA = []

    first_solutionsetB = []
    first_solutionsetC = []
    second_solutionsetA = []
    second_solutionsetC = []
    third_solutionsetA = []
    third_solutionsetB = []
    finalsolutionset = []
    solutionset = PriorityQueue() # used to ensure top 2 sol. are chosen



    #check goal equivalency 
    if goalA == goalB and goalA == goalC:
        return first_solutionsetB

    
    # Alter 2 goal equivalance to be bidirectional code
    if goalA == goalB or goalA == goalC or goalB == goalC:
        if goalA == goalB:
            start = goalA
            goal = goalC
        if goalA == goalC or goalB == goalC:
            start = goalA
            goal = goalB
        
       
        start_frontier = PriorityQueue()
        goal_frontier = PriorityQueue()
        start_visited = []
        goal_visited = []
        solutionset = []
        start_parentset = {}
        goal_parentset = {}
        start_costset = {}
        goal_costset = {}
        start_parentset[start] = None 
        goal_parentset[goal] = None  
        start_costset[start] = 0
        goal_costset[goal] = 0
        mu = float('inf')

        #if start == goal
        if start == goal:
            return solutionset


        #Append first item to the frontier
        start_frontier.append((0,start))
        goal_frontier.append((0,goal))

        while start_frontier or goal_frontier:

            #popping start frontier & checking if value is on goalset
            start_nodeset = start_frontier.pop()
            start_nodekey = start_nodeset[2]

            if start_nodekey not in start_visited:
                start_visited.append(start_nodekey)    

            if start_nodekey in (goal_parentset.keys()):
                intersection = start_nodekey


                if start_costset[intersection] + goal_costset[intersection] < mu:
                    mu = start_costset[intersection] + goal_costset[intersection]     


                if start_frontier.size() > 0:
                    start_nodecheck = start_frontier.top()
                if start_frontier.size() == 0:
                    start_nodecheck = start_nodeset

                if goal_frontier.size() > 0:
                    goal_nodecheck = goal_frontier.top()
                if goal_frontier.size() == 0:
                    goal_nodecheck = goal_nodeset

                if mu < (start_costset[start_nodecheck[2]] + goal_costset[goal_nodecheck[2]]):

                    start_visited.append(start_nodekey)
                    goal_visited.append(goal_nodekey)


                        #find crossover node list
                    crosscost = PriorityQueue()
                    crossover = set(start_parentset.keys()).intersection(set(goal_parentset.keys()))
                    for n in crossover:
                        crosscost.append((start_costset[n] + goal_costset[n],n))
                    topnode = crosscost.top()[2]

                    solutionset.append(topnode)
                    while start_parentset[topnode] is not None:
                        solutionset.append(start_parentset[topnode]) 
                        topnode = start_parentset[topnode]
                    solutionset.reverse()

                    topnode = crosscost.top()[2]
                    while goal_parentset[topnode] is not None:
                        solutionset.append(goal_parentset[topnode])
                        topnode = goal_parentset[topnode] 
                    return solutionset


            #popping goal frontier & checking if in startset
            goal_nodeset = goal_frontier.pop()
            goal_nodekey = goal_nodeset[2]

            if goal_nodekey not in goal_visited:
                goal_visited.append(goal_nodekey)



            if goal_nodekey in (start_parentset.keys()):

                intersection = goal_nodekey


                if start_costset[intersection] + goal_costset[intersection] < mu:
                    mu = start_costset[intersection] + goal_costset[intersection]     



                if start_frontier.size() > 0:
                    start_nodecheck = start_frontier.top()
                if start_frontier.size() == 0:
                    start_nodecheck = start_nodeset

                if goal_frontier.size() > 0:
                    goal_nodecheck = goal_frontier.top()
                if goal_frontier.size() == 0:
                    goal_nodecheck = goal_nodeset



                if mu < (start_costset[start_nodecheck[2]] + goal_costset[goal_nodecheck[2]]):

                    start_visited.append(start_nodekey)
                    goal_visited.append(goal_nodekey)


                        #find crossover node list
                    crosscost = PriorityQueue()
                    crossover = set(goal_parentset.keys()).intersection(set(start_parentset.keys()))
                    for n in crossover:
                        crosscost.append((start_costset[n] + goal_costset[n],n))
                    topnode = crosscost.top()[2]

                    solutionset.append(topnode)
                    while start_parentset[topnode] is not None:
                        solutionset.append(start_parentset[topnode]) 
                        topnode = start_parentset[topnode]
                    solutionset.reverse()

                    topnode = crosscost.top()[2]
                    while goal_parentset[topnode] is not None:
                        solutionset.append(goal_parentset[topnode])
                        topnode = goal_parentset[topnode]
                    return solutionset

                    ##Goal Check Code Went Here


            start_children = graph[start_nodekey].copy()

            for c in start_children:
                if c not in start_costset:
                    start_costset[c] = float("inf") 

            goal_children = graph[goal_nodekey].copy()
            for c in goal_children:
                if c not in goal_costset:
                    goal_costset[c] = float("inf") 


            #update start frontier to include child nodes
            for start_neighbor in start_children:
                if start_neighbor not in start_visited:

                    if start_costset[start_neighbor] > (start_costset[start_nodekey] + graph.get_edge_weight(start_nodekey, start_neighbor)):                    
                        start_costset[start_neighbor] = (start_costset[start_nodekey] + graph.get_edge_weight(start_nodekey, start_neighbor))
                        start_frontier.append((start_costset[start_neighbor],start_neighbor)) 
                        start_parentset[start_neighbor] = start_nodekey

            #update goal frontier to include child nodes
            for goal_neighbor in goal_children:
                if goal_neighbor not in goal_visited:                
                    if goal_costset[goal_neighbor] > (goal_costset[goal_nodekey] + graph.get_edge_weight(goal_neighbor, goal_nodekey)):                    
                        goal_costset[goal_neighbor] = (goal_costset[goal_nodekey] + graph.get_edge_weight(goal_neighbor, goal_nodekey))
                        goal_frontier.append((goal_costset[goal_neighbor],goal_neighbor)) 
                        goal_parentset[goal_neighbor] = goal_nodekey

                        

                        ##########################################################################
                        
                        
    #Normal Case: All 3 points are different goals
    
    
    
    #Add first elements to the respective fronts
    first_frontier.append((0,goalA))
    second_frontier.append((0,goalB))
    third_frontier.append((0,goalC))


    while first_frontier.size() + second_frontier.size() + third_frontier.size() > 0:
        #If all goals have been found, evaluate
        if goalAB_found and goalBC_found and goalCA_found == True:

            solution1 = solutionset.pop()[2]
            solution2 = solutionset.pop()[2]

            if solution1[0] == solution2[0]:
                solution2 = solution2[1:]
                solution1.extend(solution2)
                return solution1

            if solution1[0] == solution2[-1]:
                solution1 = solution1[1:]
                solution2.extend(solution1)
                return solution2         

            if solution2[0] == solution1[-1]:
                solution2 = solution2[1:]
                solution1.extend(solution2)
                return solution1 

            if solution1[-1] == solution2[-1]:
                solution1 = solution1[:-1]
                solution2.extend(solution1)
                return solution2


        if (goalAB_found != True) or (goalCA_found != True):
            first_front_inQ = first_frontier.pop()
            first_nodekey = first_front_inQ[2]
            first_explored.append(first_nodekey)


        #if top node is goal node (NOT CHECKING IF first node intersects with b or c explored...)
        if (first_nodekey in second_explored or first_nodekey == goalB) and goalAB_found == False:

            #find intersection of Goal A path and Goal B path, reconstruct shortest path between
            intersection = first_nodekey
            if first_costset[intersection] + second_costset[intersection] < mu_oneB:
                mu_oneB = first_costset[intersection] + second_costset[intersection]     

            if first_frontier.size() > 0:
                first_nodecheck = first_frontier.top()
            if first_frontier.size() == 0:
                first_nodecheck = first_front_inQ

            if second_frontier.size() > 0:
                second_nodecheck = second_frontier.top()
            if second_frontier.size() == 0:
                second_nodecheck = second_front_inQ

            if mu_oneB < (first_costset[first_nodecheck[2]] + second_costset[second_nodecheck[2]]):

                    #find crossover node list
                crosscost = PriorityQueue()
                crossover = set(set(first_explored)).intersection(set(second_parentset.keys()))
                for n in crossover:
                    crosscost.append((first_costset[n] + second_costset[n],n))
                topnode = crosscost.top()[2]

                first_solutionsetB.append(topnode)
                while first_parentset[topnode] != 0:
                    first_solutionsetB.append(first_parentset[topnode]) 
                    topnode = first_parentset[topnode]
                first_solutionsetB.reverse()

                topnode = crosscost.top()[2]
                while second_parentset[topnode] != 0:
                    first_solutionsetB.append(second_parentset[topnode])
                    topnode = second_parentset[topnode] 
                goalpathAB.append(first_solutionsetB)
                goalAB_found = True
                first_AB = True
                solutionset.append((crosscost.top()[0],first_solutionsetB))


        #find intersection of Goal A path and Goal C path, reconstruct shortest path between        
        if (first_nodekey in third_explored or first_nodekey == goalC) and goalCA_found == False:    
            intersection = first_nodekey
            if first_costset[intersection] + third_costset[intersection] < mu_oneC:
                mu_oneC = first_costset[intersection] + third_costset[intersection]     

            if first_frontier.size() > 0:
                first_nodecheck = first_frontier.top()
            if first_frontier.size() == 0:
                first_nodecheck = first_front_inQ

            if third_frontier.size() > 0:
                third_nodecheck = third_frontier.top()
            if third_frontier.size() == 0:
                third_nodecheck = third_front_inQ

            if mu_oneC < (first_costset[first_nodecheck[2]] + third_costset[third_nodecheck[2]]):

                    #find crossover node list
                crosscost = PriorityQueue()
                crossover = set(first_explored).intersection(set(third_parentset.keys()))
                for n in crossover:
                    crosscost.append((first_costset[n] + third_costset[n],n))
                topnode = crosscost.top()[2]

                first_solutionsetC.append(topnode)
                while first_parentset[topnode] != 0:
                    first_solutionsetC.append(first_parentset[topnode]) 
                    topnode = first_parentset[topnode]
                first_solutionsetC.reverse()

                topnode = crosscost.top()[2]
                while third_parentset[topnode] != 0:
                    first_solutionsetC.append(third_parentset[topnode])
                    topnode = third_parentset[topnode]
                first_solutionsetC.reverse()
                goalpathCA.append(first_solutionsetC)
                goalCA_found = True
                first_CA = True
                solutionset.append((crosscost.top()[0],first_solutionsetC))



        if goalAB_found and goalBC_found and goalCA_found == True:

            solution1 = solutionset.pop()[2]
            solution2 = solutionset.pop()[2]

            if solution1[0] == solution2[0]:
                solution2 = solution2[1:]
                solution1.extend(solution2)
                return solution1 

            if solution1[0] == solution2[-1]:
                solution1 = solution1[1:]
                solution2.extend(solution1)
                return solution2         

            if solution2[0] == solution1[-1]:
                solution2 = solution2[1:]
                solution1.extend(solution2)
                return solution1

            if solution1[-1] == solution2[-1]:
                solution1 = solution1[:-1]
                solution2.extend(solution1)
                return solution2         


        #popping second_fronter and getting neighbors
        if (goalAB_found != True) or (goalBC_found != True):   
            second_front_inQ = second_frontier.pop()
            second_nodekey = second_front_inQ[2]
            second_explored.append(second_nodekey)



        if (second_nodekey in first_explored or second_nodekey == goalA) and goalAB_found == False:

            #find intersection of Goal A path and Goal B path, reconstruct shortest path between
            intersection = second_nodekey
            if first_costset[intersection] + second_costset[intersection] < mu_twoA:
                mu_twoA = first_costset[intersection] + second_costset[intersection]     

            if first_frontier.size() > 0:
                first_nodecheck = first_frontier.top()
            if first_frontier.size() == 0:
                first_nodecheck = first_front_inQ

            if second_frontier.size() > 0:
                second_nodecheck = second_frontier.top()
            if second_frontier.size() == 0:
                second_nodecheck = second_front_inQ

            if mu_twoA < (first_costset[first_nodecheck[2]] + second_costset[second_nodecheck[2]]):

                    #find crossover node list
                crosscost = PriorityQueue()
                crossover = set(second_explored).intersection(set(first_parentset.keys()))
                for n in crossover:
                    crosscost.append((first_costset[n] + second_costset[n],n))
                topnode = crosscost.top()[2]

                second_solutionsetA.append(topnode)
                while second_parentset[topnode] != 0:
                    second_solutionsetA.append(second_parentset[topnode])
                    topnode = second_parentset[topnode] 
                second_solutionsetA.reverse()

                topnode = crosscost.top()[2]
                while first_parentset[topnode] != 0:
                    second_solutionsetA.append(first_parentset[topnode]) 
                    topnode = first_parentset[topnode]
                second_solutionsetA.reverse()
                goalpathAB.append(second_solutionsetA)
                solutionset.append((crosscost.top()[0],second_solutionsetA))
                goalAB_found = True
                second_AB = True


        #find intersection of Goal A path and Goal C path, reconstruct shortest path between        
        if (second_nodekey in third_explored or second_nodekey == goalC) and goalBC_found == False:    
            intersection = second_nodekey
            if second_costset[intersection] + third_costset[intersection] < mu_twoC:
                mu_twoC = second_costset[intersection] + third_costset[intersection]     

            if second_frontier.size() > 0:
                second_nodecheck = second_frontier.top()
            if second_frontier.size() == 0:
                second_nodecheck = second_front_inQ

            if third_frontier.size() > 0:
                third_nodecheck = third_frontier.top()
            if third_frontier.size() == 0:
                third_nodecheck = third_front_inQ

            if mu_twoC < (second_costset[second_nodecheck[2]] + third_costset[third_nodecheck[2]]):

                    #find crossover node list
                crosscost = PriorityQueue()
                crossover = set(second_explored).intersection(set(third_parentset.keys()))
                for n in crossover:
                    crosscost.append((second_costset[n] + third_costset[n],n))
                topnode = crosscost.top()[2]

                second_solutionsetC.append(topnode)
                while second_parentset[topnode] != 0:
                    second_solutionsetC.append(second_parentset[topnode]) 
                    topnode = second_parentset[topnode]
                second_solutionsetC.reverse()

                topnode = crosscost.top()[2]
                while third_parentset[topnode] != 0:
                    second_solutionsetC.append(third_parentset[topnode])
                    topnode = third_parentset[topnode] 
                goalpathBC.append(second_solutionsetC)
                solutionset.append((crosscost.top()[0],second_solutionsetC))
                goalBC_found = True
                second_BC = True


        if goalAB_found and goalBC_found and goalCA_found == True:

            solution1 = solutionset.pop()[2]
            solution2 = solutionset.pop()[2]

            if solution1[0] == solution2[0]:
                solution2 = solution2[1:]
                solution1.extend(solution2)
                return solution1 

            if solution1[0] == solution2[-1]:
                solution1 = solution1[1:]
                solution2.extend(solution1)
                return solution2           

            if solution2[0] == solution1[-1]:
                solution2 = solution2[1:]
                solution1.extend(solution2)
                return solution1 

            if solution1[-1] == solution2[-1]:
                solution1 = solution1[:-1]
                solution2.extend(solution1)
                return solution2


        #popping third_fronter and getting neighbors
        if (goalBC_found != True) or (goalCA_found != True):   
            third_front_inQ = third_frontier.pop()
            third_nodekey = third_front_inQ[2]
            third_explored.append(third_nodekey)


        #if top node is goal node

        if (third_nodekey in first_explored or third_nodekey == goalA) and goalCA_found == False:

            #find intersection of Goal A path and Goal B path, reconstruct shortest path between
            intersection = third_nodekey

            if first_costset[intersection] + third_costset[intersection] < mu_threeA:
                mu_threeA = first_costset[intersection] + third_costset[intersection]     

            if first_frontier.size() > 0:
                first_nodecheck = first_frontier.top()
            if first_frontier.size() == 0:
                first_nodecheck = first_front_inQ

            if third_frontier.size() > 0:
                third_nodecheck = third_frontier.top()
            if third_frontier.size() == 0:
                third_nodecheck = third_front_inQ

            if mu_threeA < (first_costset[first_nodecheck[2]] + third_costset[third_nodecheck[2]]):

                    #find crossover node list
                crosscost = PriorityQueue()
                crossover = set(third_explored).intersection(set(first_parentset.keys()))
                for n in crossover:
                    crosscost.append((first_costset[n] + third_costset[n],n))
                topnode = crosscost.top()[2]

                third_solutionsetA.append(topnode)
                while third_parentset[topnode] != 0:
                    third_solutionsetA.append(third_parentset[topnode])
                    topnode = third_parentset[topnode] 
                third_solutionsetA.reverse()

                topnode = crosscost.top()[2]
                while first_parentset[topnode] != 0:
                    third_solutionsetA.append(first_parentset[topnode]) 
                    topnode = first_parentset[topnode]
                goalpathCA.append(third_solutionsetA)
                solutionset.append((crosscost.top()[0],third_solutionsetA))
                goalCA_found = True
                third_CA = True

        if(third_nodekey in second_explored or third_nodekey == goalB) and goalBC_found == False:

            #find intersection of Goal A path and Goal B path, reconstruct shortest path between
            intersection = third_nodekey
            if third_costset[intersection] + second_costset[intersection] < mu_threeB:
                mu_threeB = third_costset[intersection] + second_costset[intersection]     

            if third_frontier.size() > 0:
                third_nodecheck = third_frontier.top()
            if third_frontier.size() == 0:
                third_nodecheck = third_front_inQ

            if second_frontier.size() > 0:
                second_nodecheck = second_frontier.top()
            if second_frontier.size() == 0:
                second_nodecheck = second_front_inQ

            if mu_threeB < (third_costset[third_nodecheck[2]] + second_costset[second_nodecheck[2]]):

                    #find crossover node list
                crosscost = PriorityQueue()
                crossover = set(third_explored).intersection(set(second_parentset.keys()))
                for n in crossover:
                    crosscost.append((third_costset[n] + second_costset[n],n))
                topnode = crosscost.top()[2]

                third_solutionsetB.append(topnode)
                while third_parentset[topnode] != 0:
                    third_solutionsetB.append(third_parentset[topnode]) 
                    topnode = third_parentset[topnode]
                third_solutionsetB.reverse()

                topnode = crosscost.top()[2]
                while second_parentset[topnode] != 0:
                    third_solutionsetB.append(second_parentset[topnode])
                    topnode = second_parentset[topnode] 
                third_solutionsetB.reverse()
                goalpathBC.append(third_solutionsetB)
                solutionset.append((crosscost.top()[0],third_solutionsetB))
                goalBC_found = True
                third_BC = True


        #Generate child nodes and add to frontier 
        if (goalAB_found != True) or (goalCA_found != True):
            first_children = graph[first_nodekey].copy()
            for c in first_children:
                if c not in first_costset:
                    first_costset[c] = float("inf") 
            for first_neighbor in first_children:
                if first_neighbor not in first_explored:
                    if first_costset[first_neighbor] > (first_costset[first_nodekey] + graph.get_edge_weight(first_nodekey, first_neighbor)):                    
                        first_costset[first_neighbor] = (first_costset[first_nodekey] + graph.get_edge_weight(first_nodekey, first_neighbor))
                        first_frontier.append((first_costset[first_neighbor],first_neighbor)) 
                        first_parentset[first_neighbor] = first_nodekey

        if (goalAB_found != True) or (goalBC_found != True):
            second_children = graph[second_nodekey].copy()
            for c in second_children:
                if c not in second_costset:
                    second_costset[c] = float("inf") 
            for second_neighbor in second_children:
                if second_neighbor not in second_explored:
                    if second_costset[second_neighbor] > (second_costset[second_nodekey] + graph.get_edge_weight(second_nodekey, second_neighbor)):                    
                        second_costset[second_neighbor] = (second_costset[second_nodekey] + graph.get_edge_weight(second_nodekey, second_neighbor))
                        second_frontier.append((second_costset[second_neighbor],second_neighbor)) 
                        second_parentset[second_neighbor] = second_nodekey

        if (goalCA_found != True) or (goalBC_found != True):
            third_children = graph[third_nodekey].copy()
            for c in third_children:
                if c not in third_costset:
                    third_costset[c] = float("inf") 

            for third_neighbor in third_children:
                if third_neighbor not in third_explored:
                    if third_costset[third_neighbor] > (third_costset[third_nodekey] + graph.get_edge_weight(third_nodekey, third_neighbor)):                    
                        third_costset[third_neighbor] = (third_costset[third_nodekey] + graph.get_edge_weight(third_nodekey, third_neighbor))
                        third_frontier.append((third_costset[third_neighbor],third_neighbor)) 
                        third_parentset[third_neighbor] = third_nodekey
    
    
    raise NotImplementedError


def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic, landmarks=None):
    """
    Exercise 4: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.
        landmarks: Iterable containing landmarks pre-computed in compute_landmarks()
            Default: None

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """

    goalA = goals[0]
    goalB = goals[1]
    goalC = goals[2]


    first_frontier = PriorityQueue() #Searches for Goal A
    second_frontier = PriorityQueue() #Searches for Goal B
    third_frontier = PriorityQueue() #Searches for Goal C

    first_explored = []
    second_explored = []
    third_explored = []

    first_parentset = {}
    first_parentset[goalA] = 0
    second_parentset = {}
    second_parentset[goalB] = 0
    third_parentset = {}
    third_parentset[goalC] = 0

    first_costset = {}
    first_costset[goalA] = 0
    second_costset = {}
    second_costset[goalB] = 0
    third_costset = {}
    third_costset[goalC] = 0

    mu_AB = float('inf')
    mu_CA = float('inf')
    mu_BC = float('inf')


    first_AB = False
    first_CA = False
    second_AB = False
    second_BC = False
    third_BC = False
    third_CA = False



    goalAB_found = False
    goalBC_found = False
    goalCA_found = False

    goalpathAB = []
    goalpathBC = []
    goalpathCA = []

    first_solutionsetB = []
    first_solutionsetC = []
    second_solutionsetA = []
    second_solutionsetC = []
    third_solutionsetA = []
    third_solutionsetB = []
    finalsolutionset = []
    solutionset = PriorityQueue() # used to ensure top 2 sol. are chosen



    #check goal equivalency 
    if goalA == goalB and goalA == goalC:
        return finalsolutionset

    if goalA == goalB or goalA == goalC or goalB == goalC:
        if goalA == goalB:
            start = goalA
            goal = goalC
        if goalA == goalC or goalB == goalC:
            start = goalA
            goal = goalB
            
            start_frontier = PriorityQueue()
            goal_frontier = PriorityQueue()
            start_visited = []
            goal_visited = []
            solutionset = []
            start_parentset = {}
            goal_parentset = {}
            start_costset = {}
            goal_costset = {}
            start_parentset[start] = 0 
            goal_parentset[goal] = 0  
            start_costset[start] = 0
            goal_costset[goal] = 0
            mu = float('inf')

            #if start == goal
            if start == goal:
                return solutionset

            #Append first item to the frontier
            start_frontier.append((0,start))
            goal_frontier.append((0,goal))

            while start_frontier or goal_frontier:

                #popping start frontier & checking if value is on goalset
                start_nodeset = start_frontier.pop()
                start_nodekey = start_nodeset[2]

                if start_nodekey != start:
                    start_costset[start_nodekey] = start_costset[start_nodekey] - euclidean_dist_heuristic(graph, start_neighbor, goal)

                if start_nodekey not in start_visited:
                    start_visited.append(start_nodekey) 

                start_children = graph[start_nodekey].copy()

                if ((start_nodekey == start) and  (goal in start_children)):
                    solutionset.append(start)
                    solutionset.append(goal)
                    return solutionset


                for c in start_children:
                    if c not in start_costset:
                        start_costset[c] = float("inf") 

                    #update start frontier to include child nodes
                for start_neighbor in start_children:
                    if start_neighbor not in start_visited:
                        start_hdis = euclidean_dist_heuristic(graph, start_neighbor, goal)
                        if start_costset[start_neighbor] > (start_costset[start_nodekey] + graph.get_edge_weight(start_nodekey, start_neighbor) + start_hdis):                    
                            start_costset[start_neighbor] = (start_costset[start_nodekey] + graph.get_edge_weight(start_nodekey, start_neighbor) + start_hdis)
                            start_frontier.append((start_costset[start_neighbor],start_neighbor)) 
                            start_parentset[start_neighbor] = start_nodekey



                if start_nodekey in goal_visited:
                    intersection = start_nodekey
                    
                    


                    if goal_parentset[intersection] != 0:
                        goal_costset[intersection] = goal_costset[goal_parentset[intersection]] + graph.get_edge_weight(goal_parentset[intersection], intersection)

                    if start_costset[intersection] + goal_costset[intersection] < mu:
                        mu = start_costset[intersection] + goal_costset[intersection]


                    if start_frontier.size() > 0:
                        start_next = start_frontier.top()
                        start_key = start_next[2]

                    if start_frontier.size() == 0:
                        start_next = start_nodeset
                        start_key = start_next[2]



                    if goal_frontier.size() > 0:
                        goal_next = goal_frontier.top()
                        goal_key = goal_next[2]


                    if goal_frontier.size() == 0:
                        goal_next = goal_nodeset
                        goal_key = goal_next[2]


                    if mu + euclidean_dist_heuristic(graph, start, goal) < (start_next[0] + goal_next[0]):

                        start_visited.append(start_nodekey)
                        goal_visited.append(goal_nodekey)


                            #find crossover node list
                        crosscost = PriorityQueue()
                        crossover = set(start_parentset.keys()).intersection(set(goal_parentset.keys()))



                        for n in crossover:
                            if n != start and n != goal:
                                crosscost.append((start_costset[start_parentset[n]] + graph.get_edge_weight(start_parentset[n], n) + goal_costset[goal_parentset[n]] + graph.get_edge_weight(goal_parentset[n], n),n))
                        topnode = crosscost.top()[2]

                        solutionset.append(topnode)
                        while start_parentset[topnode] != 0 :
                            solutionset.append(start_parentset[topnode]) 
                            topnode = start_parentset[topnode]
                        solutionset.reverse()

                        topnode = crosscost.top()[2]
                        while goal_parentset[topnode] != 0 :
                            solutionset.append(goal_parentset[topnode])
                            topnode = goal_parentset[topnode]
                        return solutionset



                #popping goal frontier & checking if in startset
                goal_nodeset = goal_frontier.pop()
                goal_nodekey = goal_nodeset[2]

                if goal_nodekey != goal:
                    goal_costset[goal_nodekey] = goal_costset[goal_parentset[goal_nodekey]] + graph.get_edge_weight(goal_parentset[goal_nodekey], goal_nodekey)

                if goal_nodekey not in goal_visited:
                    goal_visited.append(goal_nodekey)






                goal_children = graph[goal_nodekey].copy()
                for c in goal_children:
                    if c not in goal_costset:
                        goal_costset[c] = float("inf") 



                #update goal frontier to include child nodes
                for goal_neighbor in goal_children:
                    if goal_neighbor not in goal_visited:  
                        goal_hdis = euclidean_dist_heuristic(graph, goal_neighbor, start)
                        if goal_costset[goal_neighbor] > (goal_costset[goal_nodekey] + graph.get_edge_weight(goal_neighbor, goal_nodekey) + goal_hdis):                    
                            goal_costset[goal_neighbor] = (goal_costset[goal_nodekey] + graph.get_edge_weight(goal_neighbor, goal_nodekey) + goal_hdis)
                            goal_frontier.append((goal_costset[goal_neighbor],goal_neighbor)) 
                            goal_parentset[goal_neighbor] = goal_nodekey


                if goal_nodekey in start_visited:

                    intersection = goal_nodekey        

                    if start_parentset[intersection] != 0:
                        start_costset[intersection] = start_costset[start_parentset[intersection]] + graph.get_edge_weight(start_parentset[intersection], intersection)


                    if start_costset[intersection] + goal_costset[intersection] < mu:
                        mu = start_costset[intersection] + goal_costset[intersection]     

                    if start_frontier.size() > 0:
                        start_nextg = start_frontier.top()
                        start_keyg = start_nextg[2]

                    if start_frontier.size() == 0:
                        start_nextg = start_nodeset
                        start_keyg = start_nextg[2]



                    if goal_frontier.size() > 0:
                        goal_nextg = goal_frontier.top()
                        goal_keyg = goal_nextg[2]

                    if goal_frontier.size() == 0:
                        goal_nextg = goal_nodeset
                        goal_keyg = goal_nextg[2]


                    if mu + euclidean_dist_heuristic(graph, start, goal) < (start_nextg[0] + goal_nextg[0]):

                        start_visited.append(start_keyg)
                        goal_visited.append(goal_keyg)



                            #find crossover node list
                        crosscost = PriorityQueue()
                        crossover = set(goal_parentset.keys()).intersection(set(start_parentset.keys()))


                        for n in crossover:
                            if n != start and n != goal:
                                crosscost.append((start_costset[start_parentset[n]] + graph.get_edge_weight(start_parentset[n], n) + goal_costset[goal_parentset[n]] + graph.get_edge_weight(goal_parentset[n], n),n))
                        topnode = crosscost.top()[2]

                        solutionset.append(topnode)
                        while start_parentset[topnode] != 0 :
                            solutionset.append(start_parentset[topnode]) 
                            topnode = start_parentset[topnode]
                        solutionset.reverse()

                        topnode = crosscost.top()[2]
                        while goal_parentset[topnode] != 0 :
                            solutionset.append(goal_parentset[topnode])
                            topnode = goal_parentset[topnode]
                        return solutionset
        



    #Add first elements to the respective fronts
    first_frontier.append((0,goalA))
    second_frontier.append((0,goalB))
    third_frontier.append((0,goalC))


    while first_frontier.size() + second_frontier.size() + third_frontier.size() > 0:
        #If all goals have been found, evaluate
        if goalAB_found and goalBC_found and goalCA_found == True:

            solution1node = solutionset.pop()
            solution1 = solution1node[2]
            solution2node = solutionset.pop()
            solution2 = solution2node[2]
            solution3node = solutionset.pop()
            solution3 = solution3node[2]

            if solution1[0] == solution2[0]:
                solution2 = solution2[1:]
                solution1.extend(solution2)
                solution1cost = solution1node[0] + solution2node[0]
                if (goalA in solution3 and goalB in solution3 and goalC in solution3) and solution3node[0] < solution1cost:
                    return solution3          
                return solution1 

            if solution1[0] == solution2[-1]:
                solution1 = solution1[1:]
                solution2.extend(solution1)
                solution2cost = solution1node[0] + solution2node[0]
                if (goalA in solution3 and goalB in solution3 and goalC in solution3) and solution3node[0] < solution2cost:
                    return solution3   
                return solution2      

            if solution2[0] == solution1[-1]:
                solution2 = solution2[1:]
                solution1.extend(solution2)
                solution1cost = solution1node[0] + solution2node[0]
                if (goalA in solution3 and goalB in solution3 and goalC in solution3) and solution3node[0] < solution1cost:
                    return solution3
                return solution1

            if solution1[-1] == solution2[-1]:
                solution1 = solution1[:-1]
                solution2.extend(solution1)
                solution2cost = solution1node[0] + solution2node[0]
                if (goalA in solution3 and goalB in solution3 and goalC in solution3) and solution3node[0] < solution2cost:
                    return solution3 
                return solution2



        #popping first_fronter and getting neighbors

        if (goalAB_found != True) or (goalCA_found != True):
            first_front_inQ = first_frontier.pop()
            first_nodekey = first_front_inQ[2]
            first_explored.append(first_nodekey)
            if first_nodekey != goalA:
                first_costset[first_nodekey] = first_costset[first_parentset[first_nodekey]] + graph.get_edge_weight(first_parentset[first_nodekey], first_nodekey)



        #if top node is goal node (NOT CHECKING IF first node intersects with b or c explored...)
        if (first_nodekey in second_explored or first_nodekey == goalB) and goalAB_found == False:

            #find intersection of Goal A path and Goal B path, reconstruct shortest path between
            intersection = first_nodekey
            print('intersection (A-B)',first_nodekey )

            if first_costset[intersection] + second_costset[intersection] < mu_AB:
                mu_AB = first_costset[intersection] + second_costset[intersection]     

            if first_frontier.size() > 0:
                first_nodecheck = first_frontier.top()
            if first_frontier.size() == 0:
                first_nodecheck = first_front_inQ

            if second_frontier.size() > 0:
                second_nodecheck = second_frontier.top()
            if second_frontier.size() == 0:
                second_nodecheck = second_front_inQ

            if mu_AB + euclidean_dist_heuristic(graph, goalA, goalB) < (first_costset[first_nodecheck[2]] + second_costset[second_nodecheck[2]]):

                    #find crossover node list
                crosscost = PriorityQueue()
                crossover = set(set(first_explored)).intersection(set(second_parentset.keys()))
                for n in crossover:
                    if n not in second_explored:
                        second_costset[n] = second_costset[second_parentset[n]] + graph.get_edge_weight(second_parentset[n], n)
                    crosscost.append((first_costset[n] + second_costset[n],n))
                topnode = crosscost.top()[2]

                first_solutionsetB.append(topnode)
                while first_parentset[topnode] != 0:
                    first_solutionsetB.append(first_parentset[topnode]) 
                    topnode = first_parentset[topnode]
                first_solutionsetB.reverse()

                topnode = crosscost.top()[2]
                while second_parentset[topnode] != 0:
                    first_solutionsetB.append(second_parentset[topnode])
                    topnode = second_parentset[topnode] 
                goalpathAB.append(first_solutionsetB)
                goalAB_found = True
                first_AB = True
                solutionset.append((crosscost.top()[0],first_solutionsetB))
                print('solution set(A-B):',crosscost.top()[0], first_solutionsetB)


        #find intersection of Goal A path and Goal C path, reconstruct shortest path between        
        if (first_nodekey in third_explored or first_nodekey == goalC) and goalCA_found == False:    

            intersection = first_nodekey
            print('intersection (A-C):',first_nodekey)


            if first_costset[intersection] + third_costset[intersection] < mu_CA:
                mu_CA = first_costset[intersection] + third_costset[intersection]     

            if first_frontier.size() > 0:
                first_nodecheck = first_frontier.top()
            if first_frontier.size() == 0:
                first_nodecheck = first_front_inQ

            if third_frontier.size() > 0:
                third_nodecheck = third_frontier.top()
            if third_frontier.size() == 0:
                third_nodecheck = third_front_inQ

            if mu_CA + euclidean_dist_heuristic(graph, goalA, goalC) < (first_costset[first_nodecheck[2]] + third_costset[third_nodecheck[2]]):

                    #find crossover node list
                crosscost = PriorityQueue()
                crossover = set(first_explored).intersection(set(third_parentset.keys()))
                for n in crossover:
                    if n not in third_explored:
                        third_costset[n] = third_costset[third_parentset[n]] + graph.get_edge_weight(third_parentset[n], n)
                    crosscost.append((first_costset[n] + third_costset[n],n))
                topnode = crosscost.top()[2]

                first_solutionsetC.append(topnode)
                while first_parentset[topnode] != 0:
                    first_solutionsetC.append(first_parentset[topnode]) 
                    topnode = first_parentset[topnode]
                first_solutionsetC.reverse()

                topnode = crosscost.top()[2]
                while third_parentset[topnode] != 0:
                    first_solutionsetC.append(third_parentset[topnode])
                    topnode = third_parentset[topnode]
                first_solutionsetC.reverse()
                goalpathCA.append(first_solutionsetC)
                goalCA_found = True
                first_CA = True
                solutionset.append((crosscost.top()[0],first_solutionsetC))
                print('first solution set A-C:',crosscost.top()[0], first_solutionsetC )


        if goalAB_found and goalBC_found and goalCA_found == True:

            solution1node = solutionset.pop()
            solution1 = solution1node[2]
            solution2node = solutionset.pop()
            solution2 = solution2node[2]
            solution3node = solutionset.pop()
            solution3 = solution3node[2]

            if solution1[0] == solution2[0]:
                solution2 = solution2[1:]
                solution1.extend(solution2)
                solution1cost = solution1node[0] + solution2node[0]
                if (goalA in solution3 and goalB in solution3 and goalC in solution3) and solution3node[0] < solution1cost:
                    return solution3          
                return solution1 

            if solution1[0] == solution2[-1]:
                solution1 = solution1[1:]
                solution2.extend(solution1)
                solution2cost = solution1node[0] + solution2node[0]
                if (goalA in solution3 and goalB in solution3 and goalC in solution3) and solution3node[0] < solution2cost:
                    return solution3   
                return solution2      

            if solution2[0] == solution1[-1]:
                solution2 = solution2[1:]
                solution1.extend(solution2)
                solution1cost = solution1node[0] + solution2node[0]
                if (goalA in solution3 and goalB in solution3 and goalC in solution3) and solution3node[0] < solution1cost:
                    return solution3
                return solution1

            if solution1[-1] == solution2[-1]:
                solution1 = solution1[:-1]
                solution2.extend(solution1)
                solution2cost = solution1node[0] + solution2node[0]
                if (goalA in solution3 and goalB in solution3 and goalC in solution3) and solution3node[0] < solution2cost:
                    return solution3 
                return solution2


        #popping second_fronter and getting neighbors
        if (goalAB_found != True) or (goalBC_found != True):   
            second_front_inQ = second_frontier.pop()
            second_nodekey = second_front_inQ[2]
            second_explored.append(second_nodekey)
            if second_nodekey != goalB:
                second_costset[second_nodekey] = second_costset[second_parentset[second_nodekey]] + graph.get_edge_weight(second_parentset[second_nodekey], second_nodekey)

        if (second_nodekey in first_explored or second_nodekey == goalA) and goalAB_found == False:

            #find intersection of Goal A path and Goal B path, reconstruct shortest path between
            intersection = second_nodekey
            print('intersection (B-A)',second_nodekey )

            if first_costset[intersection] + second_costset[intersection] < mu_AB:
                mu_AB = first_costset[intersection] + second_costset[intersection]     

            if first_frontier.size() > 0:
                first_nodecheck = first_frontier.top()
            if first_frontier.size() == 0:
                first_nodecheck = first_front_inQ

            if second_frontier.size() > 0:
                second_nodecheck = second_frontier.top()
            if second_frontier.size() == 0:
                second_nodecheck = second_front_inQ

            if mu_AB + euclidean_dist_heuristic(graph, goalB, goalA) < (first_costset[first_nodecheck[2]] + second_costset[second_nodecheck[2]]):

                    #find crossover node list
                crosscost = PriorityQueue()
                crossover = set(second_explored).intersection(set(first_parentset.keys()))
                for n in crossover:
                    if n not in first_explored:
                        first_costset[n] = first_costset[first_parentset[n]] + graph.get_edge_weight(first_parentset[n], n)
                    crosscost.append((first_costset[n] + second_costset[n],n))
                topnode = crosscost.top()[2]

                second_solutionsetA.append(topnode)
                while second_parentset[topnode] != 0:
                    second_solutionsetA.append(second_parentset[topnode])
                    topnode = second_parentset[topnode] 
                second_solutionsetA.reverse()

                topnode = crosscost.top()[2]
                while first_parentset[topnode] != 0:
                    second_solutionsetA.append(first_parentset[topnode]) 
                    topnode = first_parentset[topnode]
                second_solutionsetA.reverse()
                goalpathAB.append(second_solutionsetA)
                solutionset.append((crosscost.top()[0],second_solutionsetA))
                print('solution set (B-A)',crosscost.top()[0],second_solutionsetA)
                goalAB_found = True
                second_AB = True

        #find intersection of Goal A path and Goal C path, reconstruct shortest path between        
        if (second_nodekey in third_explored or second_nodekey == goalC) and goalBC_found == False:    
            intersection = second_nodekey
            if second_costset[intersection] + third_costset[intersection] < mu_BC:
                mu_BC = second_costset[intersection] + third_costset[intersection]     

            if second_frontier.size() > 0:
                second_nodecheck = second_frontier.top()
            if second_frontier.size() == 0:
                second_nodecheck = second_front_inQ

            if third_frontier.size() > 0:
                third_nodecheck = third_frontier.top()
            if third_frontier.size() == 0:
                third_nodecheck = third_front_inQ

            if mu_BC + euclidean_dist_heuristic(graph, goalB, goalC) < (second_costset[second_nodecheck[2]] + third_costset[third_nodecheck[2]]):

                    #find crossover node list
                crosscost = PriorityQueue()
                crossover = set(second_explored).intersection(set(third_parentset.keys()))
                for n in crossover:
                    if n not in third_explored:
                        third_costset[n] = third_costset[third_parentset[n]] + graph.get_edge_weight(third_parentset[n], n)
                    crosscost.append((second_costset[n] + third_costset[n],n))
                topnode = crosscost.top()[2]

                second_solutionsetC.append(topnode)
                while second_parentset[topnode] != 0:
                    second_solutionsetC.append(second_parentset[topnode]) 
                    topnode = second_parentset[topnode]
                second_solutionsetC.reverse()

                topnode = crosscost.top()[2]
                while third_parentset[topnode] != 0:
                    second_solutionsetC.append(third_parentset[topnode])
                    topnode = third_parentset[topnode] 
                goalpathBC.append(second_solutionsetC)
                solutionset.append((crosscost.top()[0],second_solutionsetC))
                print('solution set (B-C):', crosscost.top()[0],second_solutionsetC)
                goalBC_found = True
                second_BC = True


        if goalAB_found and goalBC_found and goalCA_found == True:

            solution1node = solutionset.pop()
            solution1 = solution1node[2]
            solution2node = solutionset.pop()
            solution2 = solution2node[2]
            solution3node = solutionset.pop()
            solution3 = solution3node[2]

            if solution1[0] == solution2[0]:
                solution2 = solution2[1:]
                solution1.extend(solution2)
                solution1cost = solution1node[0] + solution2node[0]
                if (goalA in solution3 and goalB in solution3 and goalC in solution3) and solution3node[0] < solution1cost:
                    return solution3          
                return solution1 

            if solution1[0] == solution2[-1]:
                solution1 = solution1[1:]
                solution2.extend(solution1)
                solution2cost = solution1node[0] + solution2node[0]
                if (goalA in solution3 and goalB in solution3 and goalC in solution3) and solution3node[0] < solution2cost:
                    return solution3   
                return solution2      

            if solution2[0] == solution1[-1]:
                solution2 = solution2[1:]
                solution1.extend(solution2)
                solution1cost = solution1node[0] + solution2node[0]
                if (goalA in solution3 and goalB in solution3 and goalC in solution3) and solution3node[0] < solution1cost:
                    return solution3
                return solution1

            if solution1[-1] == solution2[-1]:
                solution1 = solution1[:-1]
                solution2.extend(solution1)
                solution2cost = solution1node[0] + solution2node[0]
                if (goalA in solution3 and goalB in solution3 and goalC in solution3) and solution3node[0] < solution2cost:
                    return solution3 
                return solution2




        #popping third_fronter and getting neighbors
        if (goalBC_found != True) or (goalCA_found != True):   
            third_front_inQ = third_frontier.pop()
            third_nodekey = third_front_inQ[2]
            third_explored.append(third_nodekey)
            if third_nodekey != goalC:
                third_costset[third_nodekey] = third_costset[third_parentset[third_nodekey]] + graph.get_edge_weight(third_parentset[third_nodekey], third_nodekey)



        #if top node is goal node

        if (third_nodekey in first_explored or third_nodekey == goalA) and goalCA_found == False:

            #find intersection of Goal A path and Goal B path, reconstruct shortest path between
            intersection = third_nodekey
            print('intersection (C-A):',third_nodekey )


            if first_costset[intersection] + third_costset[intersection] < mu_CA:
                mu_CA = first_costset[intersection] + third_costset[intersection]     

            if first_frontier.size() > 0:
                first_nodecheck = first_frontier.top()
            if first_frontier.size() == 0:
                first_nodecheck = first_front_inQ

            if third_frontier.size() > 0:
                third_nodecheck = third_frontier.top()
            if third_frontier.size() == 0:
                third_nodecheck = third_front_inQ

            if mu_CA + euclidean_dist_heuristic(graph, goalA, goalC) < (first_costset[first_nodecheck[2]] + third_costset[third_nodecheck[2]]):

                    #find crossover node list
                crosscost = PriorityQueue()
                crossover = set(third_explored).intersection(set(first_parentset.keys()))
                for n in crossover:
                    if n not in first_explored:
                        first_costset[n] = first_costset[first_parentset[n]] + graph.get_edge_weight(first_parentset[n], n)
                    crosscost.append((first_costset[n] + third_costset[n],n))
                topnode = crosscost.top()[2]

                third_solutionsetA.append(topnode)
                while third_parentset[topnode] != 0:
                    third_solutionsetA.append(third_parentset[topnode])
                    topnode = third_parentset[topnode] 
                third_solutionsetA.reverse()

                topnode = crosscost.top()[2]
                while first_parentset[topnode] != 0:
                    third_solutionsetA.append(first_parentset[topnode]) 
                    topnode = first_parentset[topnode]
                goalpathCA.append(third_solutionsetA)
                solutionset.append((crosscost.top()[0],third_solutionsetA))
                print('solution set (C-A):', crosscost.top()[0],third_solutionsetA)
                goalCA_found = True
                third_CA = True

        if(third_nodekey in second_explored or third_nodekey == goalB) and goalBC_found == False:

            #find intersection of Goal A path and Goal B path, reconstruct shortest path between
            intersection = third_nodekey
            if third_costset[intersection] + second_costset[intersection] < mu_BC:
                mu_BC = third_costset[intersection] + second_costset[intersection]     

            if third_frontier.size() > 0:
                third_nodecheck = third_frontier.top()
            if third_frontier.size() == 0:
                third_nodecheck = third_front_inQ

            if second_frontier.size() > 0:
                second_nodecheck = second_frontier.top()
            if second_frontier.size() == 0:
                second_nodecheck = second_front_inQ

            if mu_BC + euclidean_dist_heuristic(graph, goalB, goalC) < (third_costset[third_nodecheck[2]] + second_costset[second_nodecheck[2]]):

                    #find crossover node list
                crosscost = PriorityQueue()
                crossover = set(third_explored).intersection(set(second_parentset.keys()))
                for n in crossover:
                    if n not in second_explored:
                        second_costset[n] = second_costset[second_parentset[n]] + graph.get_edge_weight(second_parentset[n], n)
                    crosscost.append((third_costset[n] + second_costset[n],n))
                topnode = crosscost.top()[2]

                third_solutionsetB.append(topnode)
                while third_parentset[topnode] != 0:
                    third_solutionsetB.append(third_parentset[topnode]) 
                    topnode = third_parentset[topnode]
                third_solutionsetB.reverse()

                topnode = crosscost.top()[2]
                while second_parentset[topnode] != 0:
                    third_solutionsetB.append(second_parentset[topnode])
                    topnode = second_parentset[topnode] 
                third_solutionsetB.reverse()
                goalpathBC.append(third_solutionsetB)
                solutionset.append((crosscost.top()[0],third_solutionsetB))
                print('solutionset (C-B):', crosscost.top()[0],third_solutionsetB)
                goalBC_found = True
                third_BC = True


        #Generate child nodes and add to frontier 
        if (goalAB_found != True) or (goalCA_found != True):
            for first_neighbor in graph[first_nodekey]:
                if first_neighbor not in first_costset:
                    first_costset[first_neighbor] = float("inf") 
                    if first_neighbor not in first_explored:
                        if goalAB_found != True and goalCA_found != True:
                            first_hdisB = euclidean_dist_heuristic(graph, first_neighbor, goalB)
                            first_hdisC = euclidean_dist_heuristic(graph, first_neighbor, goalC)
                            first_hdis = min(first_hdisB, first_hdisC)
                        if goalAB_found != True and goalCA_found == True:
                            first_hdis = euclidean_dist_heuristic(graph, first_neighbor, goalB)
                        if goalAB_found == True and goalCA_found != True:
                            first_hdis = euclidean_dist_heuristic(graph, first_neighbor, goalC)
                        if first_costset[first_neighbor] > (first_costset[first_nodekey] + graph.get_edge_weight(first_nodekey, first_neighbor) + first_hdis):                    
                            first_costset[first_neighbor] = (first_costset[first_nodekey] + graph.get_edge_weight(first_nodekey, first_neighbor) + first_hdis)
                            first_frontier.append((first_costset[first_neighbor],first_neighbor)) 
                            first_parentset[first_neighbor] = first_nodekey

        if (goalAB_found != True) or (goalBC_found != True):
            for second_neighbor in graph[second_nodekey]:
                if second_neighbor not in second_costset:
                    second_costset[second_neighbor] = float("inf") 
                    if second_neighbor not in second_explored:
                        if goalAB_found != True and goalBC_found != True:
                            second_hdisA = euclidean_dist_heuristic(graph, second_neighbor, goalA)
                            second_hdisC = euclidean_dist_heuristic(graph, second_neighbor, goalC)  
                            second_hdis = min(second_hdisA, second_hdisC)
                        if goalAB_found != True and goalBC_found == True:
                            second_hdis = euclidean_dist_heuristic(graph, second_neighbor, goalA)
                        if goalAB_found == True and goalBC_found != True:
                            second_hdis = euclidean_dist_heuristic(graph, second_neighbor, goalC)
                        if second_costset[second_neighbor] > (second_costset[second_nodekey] + graph.get_edge_weight(second_nodekey, second_neighbor) + second_hdis):                    
                                second_costset[second_neighbor] = (second_costset[second_nodekey] + graph.get_edge_weight(second_nodekey, second_neighbor) + second_hdis)
                                second_frontier.append((second_costset[second_neighbor],second_neighbor)) 
                                second_parentset[second_neighbor] = second_nodekey

        if (goalCA_found != True) or (goalBC_found != True):
            for third_neighbor in graph[third_nodekey]:
                if third_neighbor not in third_costset:
                    third_costset[third_neighbor] = float("inf") 
                    if third_neighbor not in third_explored:
                        if goalCA_found != True and goalBC_found != True:
                            third_hdis = min(euclidean_dist_heuristic(graph, third_neighbor, goalA), euclidean_dist_heuristic(graph, third_neighbor, goalB))  
                        if goalCA_found == True and goalBC_found != True:
                            third_hdis = euclidean_dist_heuristic(graph, third_neighbor, goalB)
                        if goalCA_found != True and goalBC_found == True:
                            third_hdis = euclidean_dist_heuristic(graph, third_neighbor, goalA)

                        if third_costset[third_neighbor] > (third_costset[third_nodekey] + graph.get_edge_weight(third_nodekey, third_neighbor) + third_hdis):                    
                            third_costset[third_neighbor] = (third_costset[third_nodekey] + graph.get_edge_weight(third_nodekey, third_neighbor) + third_hdis)
                            third_frontier.append((third_costset[third_neighbor],third_neighbor)) 
                            third_parentset[third_neighbor] = third_nodekey



    
    
    raise NotImplementedError


def return_your_name():
    """Return your name from this function"""
    return 'Chase McGrail'
    raise NotImplementedError


def compute_landmarks(graph):
    """
    Feel free to implement this method for computing landmarks. We will call
    tridirectional_upgraded() with the object returned from this function.

    Args:
        graph (ExplorableGraph): Undirected graph to search.

    Returns:
    List with not more than 4 computed landmarks. 
    """
    return None


def custom_heuristic(graph, v, goal):
    """
       Feel free to use this method to try and work with different heuristics and come up with a better search algorithm.
       Args:
           graph (ExplorableGraph): Undirected graph to search.
           v (str): Key for the node to calculate from.
           goal (str): Key for the end node to calculate to.
       Returns:
           Custom heuristic distance between `v` node and `goal` node
       """
    pass


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    If you implement this function and submit your code to Gradescope, you'll be
    registered for the Race!

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def load_data(graph, time_left):
    """
    Feel free to implement this method. We'll call it only once 
    at the beginning of the Race, and we'll pass the output to your custom_search function.
    graph: a networkx graph
    time_left: function you can call to keep track of your remaining time.
        usage: time_left() returns the time left in milliseconds.
        the max time will be 10 minutes.

    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph.get_edge_weight(node1, node2)
    """

    # nodes = graph.nodes()
    return None


def haversine_dist_heuristic(graph, v, goal):
    """
    Note: This provided heuristic is for the Atlanta race.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Haversine distance between `v` node and `goal` node
    """

    #Load latitude and longitude coordinates in radians:
    vLatLong = (math.radians(graph.nodes[v]["pos"][0]), math.radians(graph.nodes[v]["pos"][1]))
    goalLatLong = (math.radians(graph.nodes[goal]["pos"][0]), math.radians(graph.nodes[goal]["pos"][1]))

    #Now we want to execute portions of the formula:
    constOutFront = 2*6371 #Radius of Earth is 6,371 kilometers
    term1InSqrt = (math.sin((goalLatLong[0]-vLatLong[0])/2))**2 #First term inside sqrt
    term2InSqrt = math.cos(vLatLong[0])*math.cos(goalLatLong[0])*((math.sin((goalLatLong[1]-vLatLong[1])/2))**2) #Second term
    return constOutFront*math.asin(math.sqrt(term1InSqrt+term2InSqrt)) #Straight application of formula

