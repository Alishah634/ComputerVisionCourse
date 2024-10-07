import numpy as np
import cv2
from termcolor import cprint
from skimage import io
import matplotlib.pyplot as plt 
from typing import List, Tuple
from scipy.ndimage import convolve
from scipy.optimize import least_squares
import os
import random
from tqdm import tqdm
from time import sleep
from argparse import ArgumentParser







def task1():
    cprint("Running Task 1", "white")
    # Load in the images for Task 1
    given_images = list()
    dog_image = cv2.imread('HW6_images/pics/dog_small.jpg')
    flower_image = cv2.imread('HW6_images/pics/flower_small.jpg')
    pass

def task2():
    cprint("Running Task 2", "white")
    pass

if __name__ == "__main__":
    # Create argument parser
    parser = ArgumentParser(
        description="Script to run Task 1 or Task 2. Use the arguments below to select the task.",
        epilog="Example usage: python HW6.py --task 1"
    )
    # Add arguments to select task
    parser.add_argument(
        '--task', '-t', type=int, required=True, choices=[1, 2],
        help="Select which task to run: 1 for Task 1, 2 for Task 2"
        
    )

    args = parser.parse_args()

    # Ensure you're in the correct directory
    cprint("Make sure you are running from HW6 directory!!!\n")

    # Execute the task based on the argument provided
    if args.task == 1:
        task1()
    elif args.task == 2:
        task2()
        
