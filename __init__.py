"""LR-SAF: Low-Rank Soft Attraction Fields for Line Segment Detection."""
from .saf_target import compute_saf_target, line_segment_distance
from .tnn_loss import tnn_loss
from .bounded import encode_bounded, decode_bounded
from .model import LRSAFNet, build_lr_saf
from .data import YorkUrbanSubset, collate_variable_lines
