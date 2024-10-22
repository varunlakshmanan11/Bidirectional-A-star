# Importing the required libraries.
import numpy as np
from queue import PriorityQueue
import cv2
import time

# Creating a map for the path planning.
height = 300
width = 600
Graph_map = np.ones((height, width, 3), dtype=np.uint8)*255

## Taking input from the user for start and goal nodes.
# User  input for x and y coordinates of start node.
def start_node(width, height, canvas):
    while True:
        try:
            Xs = int(input("Enter the x-coordinate of the start node(Xs): "))
            start_y = int(input("Enter the y-coordinate of the start node(Ys): "))
            Ys = height - start_y
            
            if Xs < 0 or Xs >= width or Ys < 0 or Ys >= height:
                print("The x and y coordinates of the start node is out of range.Try again!!!")
            elif np.any(canvas[Ys, Xs] != [255, 255, 255]):
                print("The x or y or both coordinates of the start node is on the obstacle.Try again!!!")
            else:
                return Xs, Ys
        except ValueError:
            print("The x and y coordinates of the start node is not a number. Try again!!!")
        

def goal_node(width, height, canvas):
    while True:
        try:
            Xg = int(input("Enter the x-coordinate of the goal node(Xg): "))
            goal_y = int(input("Enter the y-coordinate of the goal node(Yg): "))
            Yg = height - goal_y
            
            if Xg < 0 or Xg >= width or Yg < 0 or Yg >= height:
                print("The x and y coordinates of the goal node is out of range.Try again!!!")
            elif np.any(canvas[Yg,Xg] != [255,255,255]):
                print("The x or y or both coordinates of the goal node is on the obstacle.Try again!!!")
            else:
                return Xg, Yg
        except ValueError:
            print("The x and y coordinates of the goal node is not a number. Try again!!!")

# Creating a cache to store the heuristic values.
heuristic_cache = {}

for x in range(width):
    for y in range(height):
        y_transform = abs(300 - y)
    
        if ((x - 112)**2 + (y_transform - 242.5)**2 <= (40)**2):
            Graph_map[y, x] = [0,255,0]

        if ((x - 263)**2 + (y_transform - 90)**2 <= (70)**2):
            Graph_map[y, x] = [0,255,0]  

        if ((x - 445)**2 + (y_transform - 220)**2 <= (37.5)**2):
            Graph_map[y, x] = [0,255,0]  
    
# Creating a video file to store the output.
output = cv2.VideoWriter('varun_final_exam.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

## Defining 8 sets of functions to define action nodes.
# Function for moving up.
def action_up(node):
    x, y = node
    movement_up = x , y + 1
    n_x, n_y = movement_up
    return (n_x, n_y)
# Function for moving down.
def action_down(node):
    x, y = node
    movement_down = x , y - 1
    n_x, n_y = movement_down
    return (n_x, n_y)
# Function for moving left.
def action_left(node):
    x, y = node
    movement_left = x - 1 , y
    n_x, n_y = movement_left
    return (n_x, n_y)   
# Function for moving right.
def action_right(node):
    x, y = node
    movement_right = x + 1, y
    n_x, n_y = movement_right
    return (n_x, n_y)
# Function for moving upleft.
def action_up_left(node):
    x, y = node
    movement_upleft = x - 1, y + 1
    n_x, n_y = movement_upleft
    return (n_x, n_y)
#Function for moving upright.
def action_up_right(node):
    x, y = node
    movement_upright = x + 1, y + 1
    n_x, n_y = movement_upright
    return (n_x, n_y)
# Function for moving downleft.
def action_down_left(node):
    x, y = node
    movement_downleft = x - 1, y - 1
    n_x, n_y = movement_downleft
    return (n_x, n_y)
#Function for moving downright.
def action_down_right(node):
    x, y = node
    movement_downright = x + 1, y - 1
    n_x, n_y = movement_downright
    return (n_x, n_y)

# Defining a function to get possible nodes.
def possible_nodes(node):
# Setting up a dictionary to store action function as key and cost as value.
    action_set = {action_up: 1,
                action_down: 1,
                action_left: 1,
                action_right: 1,
                action_up_left: 1,
                action_up_right: 1,
                action_down_left: 1,
                action_down_right: 1}
    
    rows, columns, _ = Graph_map.shape
    next_nodes = [] # Creating a empty list for storing new nodes
    for movement, cost in action_set.items(): # For loop for getting each action and cost from the dictionary
        next_node = movement(node)
        next_x, next_y = next_node
        if 0 <= next_x < columns and 0 <= next_y < rows and np.all(Graph_map[int(next_y), int(next_x)] == [255, 255, 255]): #and not visited_check(next_node):
            if next_node not in next_nodes:
                next_nodes.append((cost, next_node)) # Adding the possible nodes to the list.
    return next_nodes


# Heuristic function
def heuristic(node, goal):
    if node in heuristic_cache:
        return heuristic_cache[node]
    else:
        heuristic_value = np.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)
        heuristic_cache[node] = heuristic_value
        return heuristic_value


# A* Algorithm
def Bi_directional_A_star(start_node, goal_node):
    forward_parent = {start_node:None}
    backward_parent = {goal_node:None}
    forward_cost_list = {start_node:0}
    backward_cost_list = {goal_node:0}
    forward_closed_list = set()
    forward_open_list = PriorityQueue()
    backward_closed_list = set()
    backward_open_list = PriorityQueue()
    forward_open_list.put(((0 + heuristic(start_node, goal_node)), start_node))
    backward_open_list.put(((0 + heuristic(goal_node, start_node)), goal_node))
    map_visualization = np.copy(Graph_map)
    step_count = 0 
    
    # While loop for checking whether both forward and backward open list is empty or not.
    while not forward_open_list.empty() and not backward_open_list.empty():
        
        # Cheking whether the forward open list is empty or not and performing the forward search from start node to common node.
        if not forward_open_list.empty():
            _, current_node = forward_open_list.get()
            forward_closed_list.add(current_node)
            
            # Checking if the current node is present in the backward closed list, if yes backtracking of the path from current node to start node is done.
            if current_node in backward_closed_list:
                path = Bidirectional_A_star_Backtracting(forward_parent, backward_parent, current_node, start_node, goal_node, map_visualization, step_count)
                for _ in range(80):
                    output.write(map_visualization)
                return path
            
            # Performing the search.   
            for cost, new_node in possible_nodes(current_node):
                cost_to_come = forward_cost_list[current_node] + cost
                if new_node not in forward_closed_list:
                    if new_node not in forward_cost_list or cost_to_come < forward_cost_list[new_node]:
                        forward_cost_list[new_node] = cost_to_come
                        forward_parent[new_node] = current_node
                        cost_total = cost_to_come + heuristic(new_node, goal_node) 
                        forward_open_list.put((cost_total, new_node))
                        cv2.arrowedLine(map_visualization, (int(current_node[0]), int(current_node[1])), (int(new_node[0]), int(new_node[1])), (0, 255, 255), 1, tipLength=0.3)
                        if step_count % 100 == 0:
                            output.write(map_visualization)
                        step_count += 1
        
        # Cheking whether the backward open list is empty or not and performing the backward search from goal node to common node.
        if not backward_open_list.empty():
            _, current_node = backward_open_list.get()
            backward_closed_list.add(current_node)
            
            # Checking if the current node is present in the forward closed list, if yes backtracking of the path from current node to start node is done.
            if current_node in forward_closed_list:
                path = Bidirectional_A_star_Backtracting(forward_parent, backward_parent, current_node, start_node, goal_node, map_visualization, step_count)
                for _ in range(80):
                    output.write(map_visualization)
                return path
            
            # Performing the search.
            for cost, new_node in possible_nodes(current_node):
                cost_to_come = backward_cost_list[current_node] + cost
                if new_node not in backward_closed_list:
                    if new_node not in backward_cost_list or cost_to_come < backward_cost_list[new_node]:
                        backward_cost_list[new_node] = cost_to_come
                        backward_parent[new_node] = current_node
                        cost_total = cost_to_come + heuristic(new_node, start_node)
                        backward_open_list.put((cost_total, new_node))
                        cv2.arrowedLine(map_visualization, (int(current_node[0]), int(current_node[1])), (int(new_node[0]), int(new_node[1])), (255, 255, 0), 1, tipLength=0.3)
                        if step_count % 100 == 0:
                            output.write(map_visualization)
                        step_count += 1

    output.release()
    return None

# Backtracking the path
def Bidirectional_A_star_Backtracting(forward_parent, backward_parent, node_intersection, start_node, goal_node, map_visualization, step_count):
    forward_path = []
    backward_path = [] 
    node1 = node_intersection
    # Backtracking the path from the intersection node to the start node.
    while node1 != start_node: 
        forward_path.append(forward_parent[node1])
        if node1 in forward_parent:
            node1 = forward_parent[node1] 
    forward_path.reverse()
    
    node2 = node_intersection
    # Backtracking the path from the intersection node to the goal node.
    while node2 != goal_node:
        backward_path.append(backward_parent[node2])
        if node2 in backward_parent:
            node2 = backward_parent[node2] 

    path = forward_path + backward_path
    
    # Drawing the path for the forward path.
    for i in range(len(forward_path) - 1):
        start_point = (int(forward_path[i][0]), int(forward_path[i][1]))  
        end_point = (int(forward_path[i + 1][0]), int(forward_path[i + 1][1]))
        cv2.arrowedLine(map_visualization, start_point, end_point, (255, 0, 0), 1, tipLength=0.3)
        if step_count % 5 == 0:
            output.write(map_visualization)
        step_count += 1 
    
    # Drawing the path for the backward path.
    for j in range(len(backward_path) - 1):
        start_point = (int(backward_path[j][0]), int(backward_path[j][1]))  
        end_point = (int(backward_path[j + 1][0]), int(backward_path[j + 1][1]))
        cv2.arrowedLine(map_visualization, start_point, end_point, (0, 0, 255), 1, tipLength=0.3)
        if step_count % 5 == 0:
            output.write(map_visualization)
        step_count += 1 
    
    cv2.imshow('Map', map_visualization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('Final_path_Varunl.png', map_visualization)
    
    return path

# Getting the start and goal nodes from the user.    
Xs, Ys = start_node(width, height, Graph_map) 
Xg, Yg = goal_node(width, height, Graph_map) 

# Initializing the start and goal nodes.
start_node = (Xs, Ys)
goal_node = (Xg, Yg)

# Starting the runtime.
start_time = time.time()   
path = Bi_directional_A_star(start_node, goal_node)

if path is None:
    print("No optimal path found")
else:
    print("Path found")

# Ending the runtime.
end_time = time.time()   
print(f'Runtime : {(end_time-start_time):.2f} seconds')
