import pandas as pd
import numpy as np
import random as rd
from typing import List, Tuple
import matplotlib.pyplot as plt
from termcolor import cprint

# Need a function for creating the plot and displaying the results: triangle filled red, user aim line is green!!!
# Need a function for creating random 3D points in repr. sapce -> maybe make 2D points then convert them
# Need a function for creating the bounds of the triangle. What ppr. does the triangle need to have ?
# Note: the user is always at the origin so that sets, and choose what the origin is, change the angle as you go.
# Try and make the graph a 180 deg positive x-axis, this is the env. 

overall_points = list()

'''
Denoting the vertices of the triangle as v1, v2, v3
v1, v2 will share thee same y co-ordinate for simplicity.
We are randomly choosing the length of the v1, v2 edge and the point v3
'''
def random_triangle() -> List[Tuple[int, int, int]]:
    while True:
        y_temp = rd.randint(5, 10)  # Ensure y is always positive and above the x-axis
        v1 = (rd.randint(-8, 8), y_temp, 1)
        v2 = (rd.randint(-8, 8), y_temp, 1)
        v3 = (rd.randint(-8, 8), rd.randint(1, 5), 1)  # Ensure v3's y-coordinate is positive
        
        # Make sure the triangle points down:
        if v3[1] > y_temp:
            continue
        
        if not are_collinear(v1, v2, v3):
            return [v1, v2, v3]

def are_collinear(v1: Tuple[int, int, int], v2: Tuple[int, int, int], v3: Tuple[int, int, int]) -> bool:
    # Calculate the determinant of the matrix formed by v1, v2, v3
    mat = np.array([v1, v2, v3])
    return np.linalg.det(mat) == 0

# Find a line starting at the origin (0,0,1) with an angle in degrees
def compute_line(aiming_angle: float = rd.randint(0,180)) -> np.ndarray:
    rad_angle = np.radians(aiming_angle)
    return cross_product(np.array([np.cos(rad_angle), np.sin(rad_angle), 0]), np.array([0, 0, 1])), aiming_angle

"""Calculate the 3D cross product of two vectors in homogeneous coordinates."""
def cross_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    return np.cross(v1, v2)

"""Get the line equation in homogeneous coordinates from two points."""
def line_from_points(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> np.ndarray:
    return cross_product(np.array(p1), np.array(p2))

"""Determine if the aiming line intersects with any of the triangle's edges."""
def find_intersection(vertices: List[Tuple[int, int, int]], line: np.ndarray) -> bool:
    for i in range(3):
        # Get the current edge vertices
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 3]
        
        # Line equation for the edge
        edge_line = line_from_points(p1, p2)
        
        # Find the intersection point using cross product
        intersection = cross_product(line, edge_line)
        overall_points.append(intersection)
        
        if intersection[2] != 0:  # To avoid division by zero, then its a point at infinity
            intersection /= intersection[2]  # Convert to Cartesian coordinates
            print(intersection)
            
            # Check if the intersection point lies within the segment
            if (min(p1[0], p2[0]) <= intersection[0] <= max(p1[0], p2[0]) and 
                min(p1[1], p2[1]) <= intersection[1] <= max(p1[1], p2[1])):
                print(intersection)
                return True
    return False

def plot_triangle_and_aim(vertices: List[Tuple[int, int, int]], line: np.ndarray, aiming_angle: float, intersects: bool):
    # Unpack vertices
    x_values = [v[0] for v in vertices] + [vertices[0][0]]
    y_values = [v[1] for v in vertices] + [vertices[0][1]]
    
    # Plot triangle
    plt.fill(x_values, y_values, 'red', alpha=0.5)
    
    # Plot aim line
    x = np.linspace(-12, 12, 100)
    y = -(line[2] + line[0] * x) / line[1]  # General form of line equation: ax + by + c = 0
    
    if intersects:
        plt.plot(x, y, 'g-', linewidth=2, label=f'Aiming line (angle={aiming_angle}°) - Intersecting')
    else:
        plt.plot(x, y, 'g--', linewidth=2, label=f'Aiming line (angle={aiming_angle}°) - Not Intersecting')

    # Origin point and direction arrow
    plt.plot(0, 0, 'ro', markersize=10)
    plt.annotate('Laser Pointer', xy=(0, 0), xytext=(1, -1),
                 arrowprops=dict(facecolor='purple', shrink=0.05),
                 fontsize=10, color='purple')
    
    # Add "Hit" or "Miss" label with a larger box
    hit_or_miss = 'Hit!' if intersects else 'Miss'
    plt.text(-10, 5, f'{hit_or_miss}', fontsize=20, color='blue',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=1"))
   
    # Plot settings
    plt.xlim(-12, 12)
    plt.ylim(-2, 12)
    plt.axhline(0, color='black', linewidth=2, label='X-axis')  # Blue X-axis
    plt.axvline(0, color='black', linewidth=2, label='Y-axis')  # Orange Y-axis
    plt.grid(True)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("Figure_1.png")
    plt.show()


# Method 2: barycentric coordinate
def calc_barycentric_coordinates(vertices: List[Tuple[int,int,int]], line: np.ndarray) -> bool: #, points = List[Tuple[int, int,int]]
    def normalize_point(point):
        if point[2] != 0:  # To avoid division by zero, then its a point at infinity
            point /= point[2]  # Convert to Cartesian coordinates
        # print(f"This is the point: {point}") # Debug
        return point
    
    def find_points(vertices: List[Tuple[int, int, int]], line: np.ndarray) -> List[Tuple[int,int,int]]:
        points = []
        for i in range(3):
            # Get the current edge vertices
            p1 = vertices[i]
            p2 = vertices[(i + 1) % 3]
            
            # Line equation for the edge
            edge_line = line_from_points(p1, p2)
            
            point = cross_product(line, edge_line)
            
            normalize_point(point)
            
            # Find the intersection point using cross product
            points.append(point)
        return points

    def check_point_location(lambda1: float, lambda2: float, lambda3: float) -> Tuple[str, bool]:
        if lambda1 > 0 and lambda2 > 0 and lambda3 > 0:
            return "Inside the triangle" , True
        elif (lambda1 == 0 and lambda2 > 0 and lambda3 > 0) or \
             (lambda2 == 0 and lambda1 > 0 and lambda3 > 0) or \
             (lambda3 == 0 and lambda1 > 0 and lambda2 > 0):
            return "On an edge of the triangle" , True 
        elif lambda1 < 0 or lambda2 < 0 or lambda3 < 0:
            return "Outside the triangle" , False
        else:
            return "Unknown condition" , False # This is a fallback, should not reached

    points = find_points(vertices, line)
    
    for point in points:
        x_1, y_1 = vertices[0][0], vertices[0][1]
        x_2, y_2 = vertices[1][0], vertices[1][1]
        x_3, y_3 = vertices[2][0], vertices[2][1]
        x, y = point[0],  point[1]
        lambda_1 = ((y_2 - y_3)*(x-x_3) + (x_3-x_2)*(y-y_3) )/( (y_2-y_3)*(x_1-x_3) + (x_3-x_2)*(y_1-y_3) )
        lambda_2 = ((y_3-y_1)*(x-x_3) + (x_1-x_3)*(y-y_3) )/( (y_2-y_3)*(x_1-x_3) + (x_3-x_2)*(y_1-y_3) )
        lambda_3 = 1 - lambda_1 - lambda_2

        print(f"Lambda 1 is: {lambda_1}, and lambda 2 is: {lambda_2} and Lambda 3 is: {lambda_3}")
        results = check_point_location(lambda_1, lambda_2,lambda_3)
        if results[1]:
            print(results[0])
            return True
    return False
        
if __name__ == '__main__':
    vertices = random_triangle()
    # vertices = [(3,5,1),(7,5,1),(5,3,1)] 
    vertices = [(1,2,1),(2,4,1),(6,4,1)] # If they intersect perfectly along one of the sides of the triangle
     
    print(f"Verticies of the Triangle:")
    [print(f"v{i}: {vertices[i-1]}") for i in range(1, 4)]  # Debug Statement
    
    # Compute the aim line
    # Calculate the angle in radians using arctan(3/4)
    angle_radians = np.arctan(2)
    # Convert the angle to degrees
    l, aiming_angle = compute_line(np.degrees(angle_radians)) #If you want to specify the angle put the angle(in deg) into the function 
    
    # Check if the aim line intersects the triangle
    print()
    print("Method 1: Checking if they are in the bounds of the triangle:")    
    if intersects := find_intersection(vertices, l):
        print("The aiming line intersects the triangle.")
        cprint("The user HIT the triangle", "green")
    else:
        print("The aiming line does not intersect the triangle.")
        cprint("The user MISSED the triangle", "red")
    
    print()
    print()
    
    # print("Method 2: Checking using Barycentric Coords:")    
    # if intersects := calc_barycentric_coordinates(vertices, l):
    #     print("The aiming line intersects the triangle.")
    #     cprint("The user HIT the triangle", "green")
    # else:
    #     print("The aiming line does not intersect the triangle.")
    #     cprint("The user MISSED the triangle", "red")
    
    # print()
    # print()

    # Plot the triangle and the aim line
    plot_triangle_and_aim(vertices, l, aiming_angle, intersects)
