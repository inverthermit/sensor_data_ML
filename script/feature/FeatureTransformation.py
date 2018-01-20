import numpy as np
import math

class FeatureTransformation():

    def scaleYZAxis(self, df):

        factor = df[['y', 'z']].max().max()
        df[['y', 'z']] = df[['y', 'z']] / factor

        return df

    def rotateYZAxis(self, df, returnList = True, rotationGranularity = 360):

        angles = np.linspace(0, 2 * math.pi, rotationGranularity + 1)[1:-1]
        result = list()

        for angle in angles:
            tmp = df
            tmp['y'] = df['y'] * math.cos(angle) + df['z'] * math.sin(angle)
            tmp['z'] = df['z'] * math.cos(angle) - df['y'] * math.sin(angle)
            result.append(tmp)

        if returnList == True:
            return result

        allData = result[0]
        for item in result[1:]:
            allData = allData.append(item)

        return allData