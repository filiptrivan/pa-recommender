"""
Outlier Detection and Removal for ALS Implicit Feedback Recommender Systems

This module provides various methods to detect and remove outliers that could negatively impact
recommendation quality, such as bot behavior, random clicking, and suspicious user patterns.
"""

import pandas as pd
from typing import Tuple, Dict
from utils.classes.Settings import Settings
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
    original_count = len(interactions)
    
    comprehensive_stats = {
        'original_count': original_count,
        'user_outliers': {},
        'interaction_outliers': {},
        'session_outliers': {},
        'final_count': 0,
        'total_removed': 0
    }
    
    # Step 1: User-level outlier detection
    interactions, user_outlier_stats = detect_user_outliers(interactions, config)
    comprehensive_stats['user_outliers'] = user_outlier_stats
    
    # Final statistics
    comprehensive_stats['final_count'] = len(interactions)
    comprehensive_stats['total_removed'] = original_count - len(interactions)
    
    return interactions, comprehensive_stats

def detect_user_outliers(interactions: pd.DataFrame, config: OutlierDetectionConfig) -> Tuple[pd.DataFrame, Dict]:
    """
    Detect user-level outliers based on interaction patterns, timing, and behavior.
    
    Args:
        interactions: DataFrame with columns [user_uid, product_id, action, created]
        config: OutlierDetectionConfig instance
        
    Returns:
        Tuple of (filtered_interactions, outlier_stats)
    """
    # Convert timestamp to datetime
    interactions = interactions.copy()
    interactions['created_dt'] = pd.to_datetime(interactions['created'], unit='s')
    interactions['date'] = interactions['created_dt'].dt.date
    
    outlier_stats = {
        'high_frequency_users': 0,
        'high_frequency_users_details': '',
        'total_outlier_users': 0
    }
    
    user_daily_counts = interactions.groupby(['user_uid', 'date']).size().reset_index(name='daily_interactions')
    high_freq_users = user_daily_counts[
        user_daily_counts['daily_interactions'] > config.max_user_interactions_per_day
    ]['user_uid'].unique()
    
    outlier_stats['high_frequency_users'] = len(high_freq_users)
    if len(high_freq_users) > 0 and Settings().ENV == 'Dev':
        outlier_stats['high_frequency_users_details'] = get_top_10_high_freq_users_details(interactions, high_freq_users, user_daily_counts, config.max_user_interactions_per_day)

    # Combine all outlier users
    all_outlier_users = set(high_freq_users)
    
    # Note: Users can appear in multiple categories, so individual counts may not sum to total
    outlier_stats['total_outlier_users'] = len(all_outlier_users)
    
    # Filter out outlier users
    outlier_mask = interactions['user_uid'].isin(all_outlier_users)
    filtered_interactions = interactions[~outlier_mask].copy()
    
    # Remove temporary columns
    filtered_interactions = filtered_interactions.drop(['created_dt', 'date'], axis=1)
    
    return filtered_interactions, outlier_stats

def get_outlier_detection_summary(stats: Dict, config: OutlierDetectionConfig) -> str:
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
        high_freq_users = user_stats.get('high_frequency_users', 0)
        summary.append(f"  High frequency users (>={config.max_user_interactions_per_day} interactions/day): {high_freq_users}")
        summary.append(user_stats['high_frequency_users_details'])
        summary.append(f"  Total outlier users: {user_stats.get('total_outlier_users', 0)}")
        summary.append("  (Note: Users can appear in multiple categories, so individual counts may not sum to total)")
    
    return "\n".join(summary)

def get_top_10_high_freq_users_details(interactions: pd.DataFrame, high_freq_users: pd.Index, user_daily_counts: pd.DataFrame, max_interactions_per_day: int):
    """
    Get the top 10 users that are filtered out due to high interaction frequency,
    along with their last 20 interactions including action type and time.
    This adds the information to the existing StringBuilder instead of sending a separate email.
    
    Args:
        interactions: DataFrame with interaction data
        high_freq_users: Index of users with high frequency interactions
        user_daily_counts: DataFrame with daily interaction counts per user
        max_interactions_per_day: Maximum allowed interactions per day
        sb: StringBuilder to append the logging information to
    """
    try:
        # Get the top 10 users by maximum daily interactions
        user_max_daily_counts = user_daily_counts.groupby('user_uid')['daily_interactions'].max().reset_index()
        top_10_users = user_max_daily_counts.nlargest(10, 'daily_interactions')
        
        if top_10_users.empty:
            return
        
        sb = StringBuilder()

        sb.append(f"\n=== TOP 10 FILTERED USERS (>={max_interactions_per_day} interactions/day) ===\n")
        sb.append(f"Total high frequency users found: {len(high_freq_users)}\n")
        sb.append(f"Showing top 10 by maximum daily interactions:\n\n")
        
        for idx, row in top_10_users.iterrows():
            user_uid = row['user_uid']
            max_daily_interactions = row['daily_interactions']
            
            sb.append(f"User: {user_uid} | Max daily interactions: {max_daily_interactions}\n")
            
            # Get last 20 interactions for this user, sorted by timestamp (newest first)
            user_interactions = interactions[interactions['user_uid'] == user_uid].copy()
            user_interactions = user_interactions.sort_values('created', ascending=False).head(20)
            
            if not user_interactions.empty:
                sb.append("Last 20 interactions:\n")
                for _, interaction in user_interactions.iterrows():
                    action = interaction['action']
                    product_id = interaction['product_id']
                    created_timestamp = interaction['created']
                    created_dt = pd.to_datetime(created_timestamp, unit='s')
                    created_iso = created_dt.isoformat()
                    
                    sb.append(f"  - Action: {action}, Product: {product_id}, Time: {created_iso}\n")
            else:
                sb.append("  No interactions found for this user.\n")
            
            sb.append("\n" + "-" * 80 + "\n\n")

        return sb.__str__()
    
    except Exception as ex: # Don't fail the main process if logging fails
        return f"Failed to log top 10 filtered users: {ex}\n"