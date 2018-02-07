

class TimeUtil:
    """input: 22:00"""
    @staticmethod
    def getMillisecondFromMinute( minute, hour=0,second = 0):
        return (hour*3600+minute*60+second)*1000
