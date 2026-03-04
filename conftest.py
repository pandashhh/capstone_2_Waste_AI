# conftest.py im Root – sorgt dafür dass pytest src/ und api/ als Module findet
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
