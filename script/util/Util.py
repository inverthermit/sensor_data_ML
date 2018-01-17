
import json


class Util:

    @staticmethod
    def getConfig(attrName):
        with open('../../config.json') as json_data_file:
            data = json.load(json_data_file)
        return data[attrName]
