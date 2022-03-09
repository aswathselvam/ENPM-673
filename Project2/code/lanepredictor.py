import cv2


import cv2

class LanePredictor:

    def __init__(self,data=None,video=None):
        self.data=data
        self.video=video

        
    def histogramEqualization(self,data):
        """
        Computes Histogram Equalization
        """
        self.data=data
        
        pass

    def detectStraightLane(self,frame):
        """
        Detects straight Lanes in a given frame
        """

        pass


    def detectCurvedLane(self):
        """
        Detects Curvatures of Lanes in a given frame
        """

        pass