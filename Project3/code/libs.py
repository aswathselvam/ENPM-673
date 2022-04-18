import cv2
import numpy as np
from utils import ImageGrid, Plotter
import math 
import matplotlib.pyplot as plt
from collections import defaultdict


from sympy import Matrix
from sympy import *


RED = (0,0,255)
GREEN = (0,255,0)
YELLOW = (0,255,255)
OLIVE = (0,128,128)
ORCHID = (204,50,153)
WHITE= (200,200,200)


class DepthComputer:

    def __init__(self,left_image=None,right_image=None):
        self.left_image = left_image
        self.right_image = right_image

    def calibrate():
        