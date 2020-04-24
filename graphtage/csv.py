import csv

from . import graphtage


def build_tree(path: str, allow_key_edits=True, *args, **kwargs):
    csv_data = []
    with open(path) as f:
        for row in csv.reader(f, *args, **kwargs):
            csv_data.append(row)
    return graphtage.build_tree(csv_data, allow_key_edits=allow_key_edits)
