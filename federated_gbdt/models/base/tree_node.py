from numba.experimental import jitclass
from numba import float32, int32, deferred_type, optional, types

# Node of decision tree, recursive model

node_type = deferred_type() # numba

array_type = types.float32[:]

spec = [ # numba
    ('feature_i', int32),
    ('threshold', optional(float32)),
    # ('value', optional(float32)),
    ('value', optional(array_type)),
    ('true_branch', optional(node_type)),
    ('false_branch', optional(node_type)),
    ('split_gain', optional(float32)),
    ('hessian_sum', optional(float32)),
    ('gradient_sum', optional(float32)),
    ('num_observations', optional(int32)),
    ('depth', optional(int32)),
]

# @jitclass(spec) # numba
class DecisionNode:
    def __init__(self, node_id="empty", feature_i=None, threshold=None,
                 value=None, true_branch=None, false_branch=None, split_gain=None, hessian_sum=None, gradient_sum=None, num_observations=None, depth=None):

        self.feature_i = feature_i          # Index for feature that is split on
        self.threshold = threshold          # Split candidate value
        self.value = value                  # Value if the node is a leaf in the tree
        self.true_branch = true_branch      #  Left subtree
        self.false_branch = false_branch    #  Right subtree
        self.node_id = node_id              # Node id for post-training stats

        # Needed for feature importance calculations
        self.split_gain = split_gain
        self.hessian_sum = hessian_sum
        self.gradient_sum = gradient_sum
        self.num_observations = num_observations
        self.depth = depth

# node_type.define(DecisionNode.class_type.instance_type) # numba