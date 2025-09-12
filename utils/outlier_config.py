"""
Configuration settings for outlier detection in ALS implicit feedback recommender systems.

This module provides different configuration presets for various use cases and environments.
"""

class OutlierDetectionConfig:
    def __init__(self):
        # User-level outlier detection
        self.max_user_interactions_per_day = 100  # Flag users with >100 interactions/day
        self.min_user_session_duration_seconds = 30  # Flag sessions <30 seconds
        self.max_user_products_per_session = 50  # Flag sessions with >50 different products
        self.user_interaction_velocity_threshold = 10  # Max interactions per minute
        
        # Statistical outlier detection
        self.z_score_threshold = 3.0  # Z-score threshold for outlier detection
        self.iqr_multiplier = 1.5  # IQR multiplier for outlier detection
        
        # Pattern-based outlier detection
        self.min_diversity_ratio = 0.1  # Min ratio of unique products to total interactions
        self.max_repeat_interaction_ratio = 0.8  # Max ratio of repeated interactions
        self.suspicious_timing_threshold = 5  # Max seconds between interactions (too fast)
        
        # Bot detection
        self.bot_like_pattern_threshold = 0.9  # Similarity threshold for bot-like patterns
        self.min_human_like_delay = 1  # Minimum seconds between interactions for human-like behavior

class EcommerceOutlierConfig(OutlierDetectionConfig):
    """Optimized for e-commerce websites with typical user behavior patterns."""
    def __init__(self):
        super().__init__()
        # E-commerce specific thresholds
        self.max_user_interactions_per_day = 150
        self.min_user_session_duration_seconds = 30
        self.max_user_products_per_session = 75
        self.user_interaction_velocity_threshold = 15
        self.z_score_threshold = 3.0
        self.iqr_multiplier = 1.5
        self.min_diversity_ratio = 0.1
        self.max_repeat_interaction_ratio = 0.8
        self.suspicious_timing_threshold = 5
        self.bot_like_pattern_threshold = 0.9
        self.min_human_like_delay = 1

def get_outlier_config(config_name: str = "default") -> OutlierDetectionConfig:
    """
    Get outlier detection configuration by name.
    
    Args:
        config_name: Name of the configuration preset
        
    Returns:
        OutlierDetectionConfig instance
        
    Available configs:
        - "default": Standard configuration
        - "conservative": Conservative outlier detection
        - "aggressive": Aggressive outlier detection
        - "ecommerce": E-commerce optimized
        - "bot_detection": Bot detection focused
        - "mobile_app": Mobile app optimized
    """
    configs = {
        "default": OutlierDetectionConfig,
        "ecommerce": EcommerceOutlierConfig,
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config name: {config_name}. Available: {list(configs.keys())}")
    
    return configs[config_name]()