

import logging

from secagg import SecAggServer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    SecAggServer().run()
