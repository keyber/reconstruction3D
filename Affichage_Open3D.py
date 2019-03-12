import numpy as np
from open3d import *    

def main():
    pcd = read_point_cloud("1a04e3eab45ca15dd86060f189eb133.points.ply") # Read the point cloud
    draw_geometries([pcd]) # Visualize the point cloud
    #print(type(pcd))     

if __name__ == "__main__":
    main()
    

