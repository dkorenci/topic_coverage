'''
Grid search functionality is moved to grid_search packege,
this module is with definitions of classes of previously
pickled data objects so that they can be still deserialized,
and the code should not be used for any other reason.
'''

# attach new moved classes as attribute of this module
# so that objects pickled before the move can be deserialized
from experiments.grid_search.engine import GridPoint, Grid
from experiments.grid_search.options import ModelOptions
import sys
current_module = sys.modules[__name__]
setattr(current_module , 'GridPoint', GridPoint)
setattr(current_module , 'Grid', Grid)
setattr(current_module , 'ModelOptions', ModelOptions)


