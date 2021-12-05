import json
import logging
import sys


class Logger:
    def __init__(self, file):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        output_file_handler = logging.FileHandler(file)
        stdout_handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(output_file_handler)
        self.logger.addHandler(stdout_handler)

    def print_and_log_j(self, content:dict, pretty=False):
        """
        print and log to a json file
        """
        if pretty:
            json_str = json.dumps(content, indent=4)
        else:
            json_str = json.dumps(content)
        self.logger.info(content)

    def print_and_log_t(self, content:dict):
        """
        print and log to a txt file
        """
        self.logger.info(content)