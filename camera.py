import pyzed.sl as sl
import numpy as np
import pandas as pd
import cv2
import statistics
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ultralytics import YOLO
 
def main():
    # --- Initialize a Camera object and open the ZED
    # Create a ZED camera object
    zed = sl.Camera()
 
    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
 
    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        exit(-1)
 
    runtime_param = sl.RuntimeParameters()

    i = 0
    image = sl.Mat()
    point_cloud = sl.Mat()
    depth_measure = sl.Mat()

    # Load YOLO Cone Model
    cone_model = YOLO('runs/detect/train6/weights/best.pt')
    times = []

    cones_df = pd.DataFrame(columns=['ConeColour', 'X', 'Y', 'Z'])

    while i < True :
        start_time = time.time()
        # Grab an image
        if zed.grab(runtime_param) == sl.ERROR_CODE.SUCCESS:
            # Display a pixel color
            zed.retrieve_image(image, sl.VIEW.LEFT)
            # zed.retrieve_image(depth, sl.VIEW.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            zed.retrieve_measure(depth_measure, sl.MEASURE.DEPTH)
            image_np = image.get_data()
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Run cone prediction on image
            results = cone_model([image_np])[0]

            fig = plt.figure(figsize=(15, 5))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            ax2.set_xlim(-5, 5)
            ax2.set_ylim(-5, 5)
            ax2.set_zlim(-5, 5)

            # Get coords of each cone in image
            for cone in results.boxes:
                x,y,w,h = cone.xywh[0]
                x = round(float(x))
                y = round(float(y))
                
                # Get the 3D point cloud values for pixel (i, j)
                cone_x, cone_y, cone_z, _ = point_cloud.get_value(x, y)[1]
                cone_class = cone_model.names[int(cone.cls)][:-5]
                ax1.scatter(x, y, c='red', s=1)
                ax2.scatter(cone_x, 0, cone_z, c=cone_class)

                # Check if cone already exists
                buffer = 0.2
                if len(cones_df[
                    (cones_df['ConeColour'] == cone_class) &
                    (cones_df['X'].between(cone_x-buffer, cone_x+buffer)) &
                    (cones_df['Z'].between(cone_z-buffer, cone_z+buffer))
                    ]) == 0 and float(depth_measure)<5:
                    cones_df.loc[len(cones_df)] = {'ConeColour': cone_class, 'X': cone_x, 'Y': 0, 'Z': cone_z}
            i = i+1
            
            ax1.imshow(image_np)
            plt.show()
        end_time = round(time.time()-start_time, 4)
        print('Step Time:', end_time)
        times.append(end_time)

    print('Average Time:', statistics.mean(times))
    print('Cone Count:', len(cones_df))
    cones_df.to_csv('cone_results.csv')
    zed.close()
    return 0
 
if __name__ == "__main__":
    main()