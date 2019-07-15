from json import load, dump
class Database:


with open(filename, 'wb') as outfile:
    json.dump(data, outfile)

    def __init__(self, path):
        self.path = path
        self.cache = {}
        self.stage = {}

    def pull(self):
        with open(self.path) as f:
            self.cache.update(load(f))

    def push(self):
        self.pull()
        self.cache.update(self.stage)
        self.stage = {}
        with open(self.path, 'wb') as f:
            dump(self.cache, self.pat)


    def __getitem__(self, key):
        if key in self.stage:
            return self.stage[key]
        elif not key in self.cache:
            self.pull()
        return self.cache[key]

    def __setitem__(self, key, value):
        self.stage[key] = value