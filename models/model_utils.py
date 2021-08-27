# attach new moved ModelOptions as attribute of this module
# so that ModelOptions pickled before the move can be deserialized
import sys

from experiments.grid_search.options import ModelOptions

current_module = sys.modules[__name__]
setattr(current_module , 'ModelOptions', ModelOptions)
