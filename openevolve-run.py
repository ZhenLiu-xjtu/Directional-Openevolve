#!/usr/bin/env python
"""
Entry point script for OpenEvolve
"""
import openevolve
print("openevolve is loaded from:", openevolve.__file__)

import sys
from openevolve.cli import main

if __name__ == "__main__":
    sys.exit(main())
