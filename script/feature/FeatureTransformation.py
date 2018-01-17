import numpy as np
import math

class FeatureTransformation():

    def scaleYZAxis(self, data):

        print('The dimensions of the input data is: ', data.shape)

        factor = np.max(data[:,1:3])
        YAndZ = data[:,1:3] / factor
        newCoor = np.column_stack((data[:,0], YAndZ))
        newData = np.column_stack((newCoor, data[:,3:]))

        print('The dimensions of the scaled data is: ', newData.shape)
        return newData

    def rotateYZAxis(self, data):

        print('The dimensions of the input data is: ', data.shape)

        data_after_rotation = data
        angels = np.linspace(0, 2 * math.pi, 361)[1:-1]

        for angel in angels:
            data_col_x = data[:,0]
            data_col_y = data[:,1]
            data_col_z = data[:,2]
            data_rest = data[:,3:]
            rotate_y = data_col_y * math.cos(angel) + data_col_z * math.sin(angel)
            ratate_z = data_col_z * math.cos(angel) - data_col_y * math.sin(angel)
            new_y_z = np.column_stack((rotate_y, ratate_z))
            coor_after_rotation = np.column_stack((data_col_x, new_y_z))
            rotated_data = np.column_stack((coor_after_rotation, data_rest))
            data_after_rotation = np.vstack((data_after_rotation, rotated_data))

        print('The dimensions of the rotated data is: ', data_after_rotation.shape)
        return data_after_rotation
