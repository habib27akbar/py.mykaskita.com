# passenger_wsgi.py  (fixed)
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from app import app as application  # <-- export "application"
