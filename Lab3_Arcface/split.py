import splitfolders
import os

path = os.path.dirname(__file__)
splitfolders.ratio(
    input=os.path.join(path, 'data/EuroSAT_RGB'),
    output=os.path.join(path, 'data/split'),
    ratio=(0.9, 0.1)
)