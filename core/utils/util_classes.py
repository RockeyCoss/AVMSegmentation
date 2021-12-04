class Logger:
    def __init__(self, file):
        self.file = file

    def print_and_log(self, content):
        with open(self.file, 'a') as f:
            print(content, file=f)