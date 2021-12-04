import json


class Logger:
    def __init__(self, file):
        self.file = file

    def print_and_log(self, content:dict, pretty=False):
        with open(self.file, 'a') as f:
            if pretty:
                json_str = json.dumps(content, indent=4)
            else:
                json_str = json.dumps(content)
            print(json_str, file=f)