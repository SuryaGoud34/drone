import numpy as np
from filterpy.kalman import KalmanFilter

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2-xx1)
    h = np.maximum(0., yy2-yy1)
    wh = w*h
    return wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
        + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)

class Tracker:
    count = 0
    def __init__(self,bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        self.kf.H = np.zeros((4,7))
        self.kf.H[:4,:4]=np.eye(4)
        self.kf.x[:4]=bbox.reshape((4,1))
        self.time_since_update=0
        self.id=Tracker.count
        Tracker.count+=1

    def update(self,bbox):
        self.time_since_update=0
        self.kf.update(bbox)

    def predict(self):
        self.kf.predict()
        self.time_since_update+=1
        return self.kf.x[:4].reshape((4,))
