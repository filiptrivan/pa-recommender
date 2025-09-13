"""
Outlier Detection and Removal for ALS Implicit Feedback Recommender Systems

This module provides various methods to detect and remove outliers that could negatively impact
recommendation quality, such as bot behavior, random clicking, and suspicious user patterns.
"""

import pandas as pd
from typing import Tuple, Dict
from utils.classes.StringBuilder import StringBuilder
from utils.outlier_config import OutlierDetectionConfig

def outlier_detection(interactions: pd.DataFrame, config: OutlierDetectionConfig) -> Tuple[pd.DataFrame, Dict]:
    """
    Perform comprehensive outlier detection using all available methods.
    
    Args:
        interactions: DataFrame with interaction data
        config: OutlierDetectionConfig instance
        
    Returns:
        Tuple of (filtered_interactions, comprehensive_outlier_stats)
    """
    sb = StringBuilder()
    original_count = len(interactions)
    sb.append("OUTLIER DETECTION\n")
    sb.append(f"Original interactions: {original_count}\n")
    
    sb.append("Using standard processing\n\n")
    
    comprehensive_stats = {
        'original_count': original_count,
        'user_outliers': {},
        'interaction_outliers': {},
        'session_outliers': {},
        'final_count': 0,
        'total_removed': 0
    }
    
    # Step 1: User-level outlier detection
    sb.append("1. USER-LEVEL OUTLIER DETECTION\n")
    interactions, user_outlier_stats = detect_user_outliers(interactions, config)
    comprehensive_stats['user_outliers'] = user_outlier_stats
    sb.append(f"After user outlier removal: {len(interactions)} interactions\n\n")
    
    # Final statistics
    comprehensive_stats['final_count'] = len(interactions)
    comprehensive_stats['total_removed'] = original_count - len(interactions)
    
    sb.append("=== OUTLIER DETECTION SUMMARY ===\n")
    sb.append(f"Original interactions: {original_count}\n")
    sb.append(f"Final interactions: {len(interactions)}\n")
    sb.append(f"Total removed: {comprehensive_stats['total_removed']}\n")
    sb.append(f"Removal rate: {comprehensive_stats['total_removed']/original_count*100:.2f}%\n")
    
    return interactions, comprehensive_stats

def detect_user_outliers(interactions: pd.DataFrame, config: OutlierDetectionConfig) -> Tuple[pd.DataFrame, Dict]:
    """
    Detect user-level outliers based on interaction patterns, timing, and behavior.
    Optimized for big data with vectorized operations and optional parallel processing.
    
    Args:
        interactions: DataFrame with columns [user_uid, product_id, action, created]
        config: OutlierDetectionConfig instance
        use_parallel: Whether to use parallel processing for user analysis
        n_jobs: Number of parallel jobs (None for auto-detection)
        
    Returns:
        Tuple of (filtered_interactions, outlier_stats)
    """
    sb = StringBuilder()
    original_count = len(interactions)
    sb.append(f"Original interactions: {original_count}\n")
    
    # Convert timestamp to datetime
    interactions = interactions.copy()
    interactions['created_dt'] = pd.to_datetime(interactions['created'], unit='s')
    interactions['date'] = interactions['created_dt'].dt.date
    
    outlier_stats = {
        'high_frequency_users': 0,
        'total_outlier_users': 0
    }
    
    user_daily_counts = interactions.groupby(['user_uid', 'date']).size().reset_index(name='daily_interactions')
    high_freq_users = user_daily_counts[
        user_daily_counts['daily_interactions'] > config.max_user_interactions_per_day
    ]['user_uid'].unique()
    
    outlier_stats['high_frequency_users'] = len(high_freq_users)
    sb.append(f"High frequency users (>={config.max_user_interactions_per_day} interactions/day): {len(high_freq_users)}\n")
    
    # Combine all outlier users
    all_outlier_users = set(high_freq_users)
    
    # Update outlier stats with correct counts
    # Note: Users can appear in multiple categories, so individual counts may not sum to total
    outlier_stats['total_outlier_users'] = len(all_outlier_users)
    sb.append(f"Total outlier users: {len(all_outlier_users)}\n")
    
    # Filter out outlier users
    outlier_mask = interactions['user_uid'].isin(all_outlier_users)
    filtered_interactions = interactions[~outlier_mask].copy()
    
    # Remove temporary columns
    filtered_interactions = filtered_interactions.drop(['created_dt', 'date'], axis=1)
    
    return filtered_interactions, outlier_stats

def get_outlier_detection_summary(stats: Dict) -> str:
    """Generate a human-readable summary of outlier detection results."""
    summary = []
    summary.append("Outlier Detection Summary:")
    summary.append(f"  Original interactions: {stats['original_count']:,}")
    summary.append(f"  Final interactions: {stats['final_count']:,}")
    summary.append(f"  Total removed: {stats['total_removed']:,}")
    summary.append(f"  Removal rate: {stats['total_removed']/stats['original_count']*100:.2f}%")
    
    if 'user_outliers' in stats:
        user_stats = stats['user_outliers']
        summary.append("\nUser-level outliers:")
        summary.append(f"  High frequency users: {user_stats.get('high_frequency_users', 0)}")
        summary.append(f"  Total outlier users: {user_stats.get('total_outlier_users', 0)}")
        summary.append("  (Note: Users can appear in multiple categories, so individual counts may not sum to total)")
    
    return "\n".join(summary)