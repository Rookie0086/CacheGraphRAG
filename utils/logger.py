import datetime as dt
import json
import os
from collections import defaultdict

from utils.base import create_dir


class Logger:

    def __init__(self, log_dir="./", log_name="log.txt"):
        # self.log_root = os.path.join(os.path.dirname(__file__), '..', log_dir)
        # os.makedirs(self.log_root, exist_ok=True)
        create_dir(log_dir)
        self.log_name = os.path.join(log_dir, log_name)
        self.info_dict = defaultdict(list)

    def update(self, info):
        self.info_dict.update(info)

    def add(self, key, value):
        self.info_dict[key].append(value)

    def get_times(self, key):
        return self.info_dict.get(key, None)

    def all_info(self):
        return self.info_dict

    def log(self, *args, oneline=False):
        head = f"{dt.datetime.now().time()} "
        tail = "\r" if oneline else "\n"
        the_whole_line = head + " ".join(map(str, args)) + tail
        print(the_whole_line, end="", flush=True)
        with open(self.log_name, "a+") as f:
            print(the_whole_line, end="", file=f, flush=True)

    def save(self, log_name=None):
        if not log_name:
            log_name = self.log_name
        with open(log_name, "w") as file:
            json.dump(self.info_dict, file, indent=2)
        print("log save to", log_name)


if __name__ == "__main__":
    logger = Logger()
    logger.log("This is a test log.")
    import time

    from utils.timer import Timer

    timer = Timer()

    with timer.timing("update"):
        logger.log("Updating progress:", "25%", oneline=True)
        time.sleep(1)

    with timer.timing("update"):
        logger.log("Updating progress:", "75%", oneline=True)
        time.sleep(1)

    with timer.timing("update"):
        logger.log("Updating progress:", "100%", oneline=False)
        time.sleep(2)

    logger.log("Finished logging.")
    logger.log("\n")
    logger.log(timer.summary())

    logger.add("hhhe", "123")

    # print(logger.all_info())
