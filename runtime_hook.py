import sys
import os

# Point Python to the bundled packages at runtime
resources = os.path.join(os.path.dirname(sys.executable), '..', 'Resources')
resources = os.path.abspath(resources)
if resources not in sys.path:
    sys.path.insert(0, resources)