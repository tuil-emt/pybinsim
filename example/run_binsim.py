import pybinsim
import logging

pybinsim.logger.setLevel(logging.DEBUG)    # defaults to INFO

with pybinsim.BinSim('pyBinSimSettings_isoperare.txt') as binsim:
    binsim.stream_start()