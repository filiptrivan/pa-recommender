"""
Configuration settings for outlier detection in ALS implicit feedback recommender systems.

This module provides different configuration presets for various use cases and environments.
"""

class OutlierDetectionConfig:
    def __init__(self):
        # User-level outlier detection
        self.max_user_interactions_per_day = 150  # Flag users with >150 interactions/day     