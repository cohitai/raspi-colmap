from reconstruction import DockerizedColmap
import logging
import sys

# create logger with 'spam_application'
logging.getLogger('Raspi- application')
logging.basicConfig(stream=sys.stdout, filemode='a', level=logging.DEBUG)

a = DockerizedColmap("/home/liteandfog/raspi-colmap/d1", "/home/liteandfog/raspi-colmap/d2", \
                     "/home/liteandfog/raspi-colmap/d3", outliers=True, poisson=True)
a.reconstruct()