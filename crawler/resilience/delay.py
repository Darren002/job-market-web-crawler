import time
import random


def human_delay():

    delay = random.uniform(1.5, 4)

    print(f"Sleeping for {delay:.2f} seconds")

    time.sleep(delay)