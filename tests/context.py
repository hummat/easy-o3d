"""Helper module for importing the package without installation."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from easy_o3d import utils, registration, interfaces
from scripts import run_registration, hyperopt
