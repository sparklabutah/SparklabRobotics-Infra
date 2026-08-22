"""Web harness: a high-level agent in front of the low-level VLA.

``server.py`` serves the UI and proxies a running ``rollout_live`` control port;
``agents.py`` holds the planners. See ``server.py``'s docstring for the process
split and for driving it from an external agent instead of the built-in loop.
"""
