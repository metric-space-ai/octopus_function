import os

class Template:
    def __init__(self):
        self.file = "model.dat"
        self.model = None
        self.results = {}

    def get_result(self, id):
        return self.results[id]

    def set_result(self, id, response):
        self.results[id] = response

    def setup(self):
        f = open(self.file, "a")
        f.write("Lets create some model file")
        f.close()

    def setup_condition(self) -> bool:
        if not os.path.isfile(self.file):
            return True

        return False

    def warmup(self):
        if self.model == None:
            self.model = True

    def warmup_condition(self) -> bool:
        if self.model == None:
            return True

        return False
