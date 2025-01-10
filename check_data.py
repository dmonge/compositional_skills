"""
Check the data.
"""

with open('data/tasks.txt') as file:
    lines = file.readlines()
    assert len(lines) == len(set(lines))  # no duplicates
