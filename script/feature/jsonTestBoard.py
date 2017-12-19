import json

try:
    from types import SimpleNamespace as Namespace
except ImportError:
    # Python 2.x fallback
    from argparse import Namespace

data2 = '{"name": "John Smith", "hometown": {"name": "New York", "id": 123}}'
data = '{"timestamp":1512600656276,"event":{"variable":"acceleration","content":[-1.1786659955978394,-0.05114848166704178,0.08555429428815842,1510837950540]}}'
x = json.loads(data, object_hook=lambda d: Namespace(**d))

print (x.timestamp, x.event.variable, x.event.content[0])
