"""
Outlier Detection and Removal for ALS Implicit Feedback Recommender Systems

This module provides various methods to detect and remove outliers that could negatively impact
recommendation quality, such as bot behavior, random clicking, and suspicious user patterns.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix
from typing import Tuple, List, Dict, Optional
from collections import defaultdict
import logging
from utils.classes.StringBuilder import StringBuilder
from utils.outlier_config import OutlierDetectionConfig

logger = logging.getLogger(__name__)

def detect_user_outliers(interactions: pd.DataFrame, config: OutlierDetectionConfig = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Detect user-level outliers based on interaction patterns, timing, and behavior.
    
    Args:
        interactions: DataFrame with columns [user_uid, product_id, action, created]
        config: OutlierDetectionConfig instance
        
    Returns:
        Tuple of (filtered_interactions, outlier_stats)
    """
    if config is None:
        config = OutlierDetectionConfig()
    
    sb = StringBuilder()
    original_count = len(interactions)
    sb.append(f"Original interactions: {original_count}\n")
    
    # Convert timestamp to datetime
    interactions = interactions.copy()
    interactions['created_dt'] = pd.to_datetime(interactions['created'], unit='s')
    interactions['date'] = interactions['created_dt'].dt.date
    
    outlier_stats = {
        'high_frequency_users': 0,
        'bot_like_users': 0,
        'low_diversity_users': 0,
        'suspicious_timing_users': 0,
        'total_outlier_users': 0
    }
    
    # 1. Detect high-frequency users (potential bots)
    user_daily_counts = interactions.groupby(['user_uid', 'date']).size().reset_index(name='daily_interactions')
    high_freq_users = user_daily_counts[
        user_daily_counts['daily_interactions'] > config.max_user_interactions_per_day
    ]['user_uid'].unique()
    
    outlier_stats['high_frequency_users'] = len(high_freq_users)
    sb.append(f"High frequency users (>={config.max_user_interactions_per_day} interactions/day): {len(high_freq_users)}\n")
    
    # 2. Detect users with suspicious interaction patterns
    suspicious_users = set()
    
    for user_id in interactions['user_uid'].unique():
        user_interactions = interactions[interactions['user_uid'] == user_id].sort_values('created')
        
        # Check for bot-like patterns
        if _is_bot_like_user(user_interactions, config):
            suspicious_users.add(user_id)
            outlier_stats['bot_like_users'] += 1
        
        # Check for low diversity (clicking same products repeatedly)
        if _has_low_diversity(user_interactions, config):
            suspicious_users.add(user_id)
            outlier_stats['low_diversity_users'] += 1
        
        # Check for suspicious timing patterns
        if _has_suspicious_timing(user_interactions, config):
            suspicious_users.add(user_id)
            outlier_stats['suspicious_timing_users'] += 1
    
    # Combine all outlier users
    all_outlier_users = set(high_freq_users) | suspicious_users
    outlier_stats['total_outlier_users'] = len(all_outlier_users)
    
    # Filter out outlier users
    filtered_interactions = interactions[~interactions['user_uid'].isin(all_outlier_users)].copy()
    filtered_count = len(filtered_interactions)
    
    sb.append(f"Filtered interactions: {filtered_count}\n")
    sb.append(f"Interactions removed: {original_count - filtered_count}\n")
    sb.append(f"Outlier users removed: {len(all_outlier_users)}\n")
    
    # Remove temporary columns
    filtered_interactions = filtered_interactions.drop(['created_dt', 'date'], axis=1)
    
    return filtered_interactions, outlier_stats

def _is_bot_like_user(user_interactions: pd.DataFrame, config: OutlierDetectionConfig) -> bool:
    """Check if user exhibits bot-like behavior patterns."""
    if len(user_interactions) < 10:  # Need minimum interactions to detect patterns
        return False
    
    # Check interaction velocity (interactions per minute)
    time_diffs = user_interactions['created'].diff().dropna()
    if len(time_diffs) > 0:
        avg_seconds_between = time_diffs.mean()
        interactions_per_minute = 60 / avg_seconds_between if avg_seconds_between > 0 else float('inf')
        
        if interactions_per_minute > config.user_interaction_velocity_threshold:
            return True
    
    # Check for extremely regular timing patterns (bot-like)
    if len(time_diffs) > 5:
        time_diffs_std = time_diffs.std()
        time_diffs_mean = time_diffs.mean()
        cv = time_diffs_std / time_diffs_mean if time_diffs_mean > 0 else 0
        
        # Very low coefficient of variation suggests automated behavior
        if cv < 0.1 and time_diffs_mean < 10:  # Very regular, fast interactions
            return True
    
    return False

def _has_low_diversity(user_interactions: pd.DataFrame, config: OutlierDetectionConfig) -> bool:
    """Check if user has low product diversity (clicking same products repeatedly)."""
    if len(user_interactions) < 5:
        return False
    
    unique_products = user_interactions['product_id'].nunique()
    total_interactions = len(user_interactions)
    diversity_ratio = unique_products / total_interactions
    
    return diversity_ratio < config.min_diversity_ratio

def _has_suspicious_timing(user_interactions: pd.DataFrame, config: OutlierDetectionConfig) -> bool:
    """Check for suspicious timing patterns (too fast, too regular)."""
    if len(user_interactions) < 5:
        return False
    
    time_diffs = user_interactions['created'].diff().dropna()
    
    # Check for too many very fast interactions
    very_fast_interactions = (time_diffs < config.suspicious_timing_threshold).sum()
    if very_fast_interactions / len(time_diffs) > 0.8:  # 80% of interactions are too fast
        return True
    
    # Check for too many interactions with identical timing
    if len(time_diffs) > 0:
        most_common_diff = time_diffs.mode().iloc[0] if not time_diffs.mode().empty else 0
        identical_timing_count = (time_diffs == most_common_diff).sum()
        if identical_timing_count / len(time_diffs) > 0.7:  # 70% have identical timing
            return True
    
    return False

def detect_interaction_outliers(interactions: pd.DataFrame, config: OutlierDetectionConfig = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Detect and remove outlier interactions using statistical methods.
    
    Args:
        interactions: DataFrame with interaction data
        config: OutlierDetectionConfig instance
        
    Returns:
        Tuple of (filtered_interactions, outlier_stats)
    """
    if config is None:
        config = OutlierDetectionConfig()
    
    sb = StringBuilder()
    original_count = len(interactions)
    sb.append(f"Original interactions: {original_count}\n")
    
    outlier_stats = {
        'z_score_outliers': 0,
        'iqr_outliers': 0,
        'total_outlier_interactions': 0
    }
    
    # Calculate interaction weights for outlier detection
    interactions = interactions.copy()
    interactions['weight'] = interactions['action'].map({
        'purchase': 1.0,
        'initiate_checkout': 0.7,
        'add_to_cart': 0.5,
        'add_to_wishlist': 0.3,
        'content_view': 0.1
    })
    
    # 1. Z-score based outlier detection
    z_scores = np.abs(stats.zscore(interactions['weight']))
    z_score_outliers = interactions[z_scores > config.z_score_threshold]
    outlier_stats['z_score_outliers'] = len(z_score_outliers)
    
    # 2. IQR based outlier detection
    Q1 = interactions['weight'].quantile(0.25)
    Q3 = interactions['weight'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - config.iqr_multiplier * IQR
    upper_bound = Q3 + config.iqr_multiplier * IQR
    
    iqr_outliers = interactions[
        (interactions['weight'] < lower_bound) | 
        (interactions['weight'] > upper_bound)
    ]
    outlier_stats['iqr_outliers'] = len(iqr_outliers)
    
    # Combine outlier detection methods
    outlier_indices = set(z_score_outliers.index) | set(iqr_outliers.index)
    outlier_stats['total_outlier_interactions'] = len(outlier_indices)
    
    # Filter out outliers
    filtered_interactions = interactions[~interactions.index.isin(outlier_indices)].copy()
    filtered_count = len(filtered_interactions)
    
    sb.append(f"Z-score outliers: {outlier_stats['z_score_outliers']}\n")
    sb.append(f"IQR outliers: {outlier_stats['iqr_outliers']}\n")
    sb.append(f"Total outlier interactions: {outlier_stats['total_outlier_interactions']}\n")
    sb.append(f"Filtered interactions: {filtered_count}\n")
    sb.append(f"Interactions removed: {original_count - filtered_count}\n")
    
    return filtered_interactions, outlier_stats

def detect_session_outliers(interactions: pd.DataFrame, config: OutlierDetectionConfig = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Detect outlier sessions based on session characteristics.
    
    Args:
        interactions: DataFrame with interaction data
        config: OutlierDetectionConfig instance
        
    Returns:
        Tuple of (filtered_interactions, outlier_stats)
    """
    if config is None:
        config = OutlierDetectionConfig()
    
    sb = StringBuilder()
    original_count = len(interactions)
    sb.append(f"Original interactions: {original_count}\n")
    
    # Convert timestamp to datetime
    interactions = interactions.copy()
    interactions['created_dt'] = pd.to_datetime(interactions['created'], unit='s')
    
    outlier_stats = {
        'short_sessions': 0,
        'high_product_sessions': 0,
        'total_outlier_sessions': 0
    }
    
    # Group interactions by user and session (assuming 30-minute session timeout)
    interactions['session_id'] = (
        interactions.groupby('user_uid')['created_dt']
        .transform(lambda x: (x.diff() > pd.Timedelta(minutes=30)).cumsum())
    )
    
    session_stats = interactions.groupby(['user_uid', 'session_id']).agg({
        'created_dt': ['min', 'max', 'count'],
        'product_id': 'nunique'
    }).reset_index()
    
    session_stats.columns = ['user_uid', 'session_id', 'session_start', 'session_end', 'interaction_count', 'unique_products']
    session_stats['session_duration'] = (session_stats['session_end'] - session_stats['session_start']).dt.total_seconds()
    
    # Detect outlier sessions
    outlier_sessions = []
    
    # Short sessions (potential bot behavior)
    short_sessions = session_stats[session_stats['session_duration'] < config.min_user_session_duration_seconds]
    outlier_sessions.extend(short_sessions[['user_uid', 'session_id']].values.tolist())
    outlier_stats['short_sessions'] = len(short_sessions)
    
    # Sessions with too many different products (potential random clicking)
    high_product_sessions = session_stats[session_stats['unique_products'] > config.max_user_products_per_session]
    outlier_sessions.extend(high_product_sessions[['user_uid', 'session_id']].values.tolist())
    outlier_stats['high_product_sessions'] = len(high_product_sessions)
    
    # Remove duplicate sessions
    outlier_sessions = list(set(tuple(session) for session in outlier_sessions))
    outlier_stats['total_outlier_sessions'] = len(outlier_sessions)
    
    # Filter out outlier sessions
    outlier_mask = interactions.apply(
        lambda row: (row['user_uid'], row['session_id']) in outlier_sessions, 
        axis=1
    )
    
    filtered_interactions = interactions[~outlier_mask].copy()
    filtered_count = len(filtered_interactions)
    
    sb.append(f"Short sessions (<{config.min_user_session_duration_seconds}s): {outlier_stats['short_sessions']}\n")
    sb.append(f"High product sessions (>{config.max_user_products_per_session} products): {outlier_stats['high_product_sessions']}\n")
    sb.append(f"Total outlier sessions: {outlier_stats['total_outlier_sessions']}\n")
    sb.append(f"Filtered interactions: {filtered_count}\n")
    sb.append(f"Interactions removed: {original_count - filtered_count}\n")
    
    # Remove temporary columns
    filtered_interactions = filtered_interactions.drop(['created_dt', 'session_id'], axis=1)
    
    return filtered_interactions, outlier_stats

def comprehensive_outlier_detection(interactions: pd.DataFrame, config: OutlierDetectionConfig = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Perform comprehensive outlier detection using all available methods.
    
    Args:
        interactions: DataFrame with interaction data
        config: OutlierDetectionConfig instance
        
    Returns:
        Tuple of (filtered_interactions, comprehensive_outlier_stats)
    """
    if config is None:
        config = OutlierDetectionConfig()
    
    sb = StringBuilder()
    original_count = len(interactions)
    sb.append("=== COMPREHENSIVE OUTLIER DETECTION ===\n")
    sb.append(f"Original interactions: {original_count}\n\n")
    
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
    
    # Step 2: Session-level outlier detection
    sb.append("2. SESSION-LEVEL OUTLIER DETECTION\n")
    interactions, session_outlier_stats = detect_session_outliers(interactions, config)
    comprehensive_stats['session_outliers'] = session_outlier_stats
    sb.append(f"After session outlier removal: {len(interactions)} interactions\n\n")
    
    # Step 3: Interaction-level outlier detection
    sb.append("3. INTERACTION-LEVEL OUTLIER DETECTION\n")
    interactions, interaction_outlier_stats = detect_interaction_outliers(interactions, config)
    comprehensive_stats['interaction_outliers'] = interaction_outlier_stats
    sb.append(f"After interaction outlier removal: {len(interactions)} interactions\n\n")
    
    # Final statistics
    comprehensive_stats['final_count'] = len(interactions)
    comprehensive_stats['total_removed'] = original_count - len(interactions)
    
    sb.append("=== OUTLIER DETECTION SUMMARY ===\n")
    sb.append(f"Original interactions: {original_count}\n")
    sb.append(f"Final interactions: {len(interactions)}\n")
    sb.append(f"Total removed: {comprehensive_stats['total_removed']}\n")
    sb.append(f"Removal rate: {comprehensive_stats['total_removed']/original_count*100:.2f}%\n")
    
    return interactions, comprehensive_stats

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
        summary.append(f"  Bot-like users: {user_stats.get('bot_like_users', 0)}")
        summary.append(f"  Low diversity users: {user_stats.get('low_diversity_users', 0)}")
        summary.append(f"  Suspicious timing users: {user_stats.get('suspicious_timing_users', 0)}")
        summary.append(f"  Total outlier users: {user_stats.get('total_outlier_users', 0)}")
    
    if 'session_outliers' in stats:
        session_stats = stats['session_outliers']
        summary.append("\nSession-level outliers:")
        summary.append(f"  Short sessions: {session_stats.get('short_sessions', 0)}")
        summary.append(f"  High product sessions: {session_stats.get('high_product_sessions', 0)}")
        summary.append(f"  Total outlier sessions: {session_stats.get('total_outlier_sessions', 0)}")
    
    if 'interaction_outliers' in stats:
        interaction_stats = stats['interaction_outliers']
        summary.append("\nInteraction-level outliers:")
        summary.append(f"  Z-score outliers: {interaction_stats.get('z_score_outliers', 0)}")
        summary.append(f"  IQR outliers: {interaction_stats.get('iqr_outliers', 0)}")
        summary.append(f"  Total outlier interactions: {interaction_stats.get('total_outlier_interactions', 0)}")
    
    return "\n".join(summary)