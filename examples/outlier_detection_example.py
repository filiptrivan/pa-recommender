"""
Example script demonstrating how to use outlier detection in ALS implicit feedback recommender systems.

This script shows different ways to configure and use outlier detection for various scenarios.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath("../.."))

from utils.outlier_detection import comprehensive_outlier_detection, detect_user_outliers, detect_interaction_outliers, detect_session_outliers
from utils.outlier_config import get_outlier_config, create_custom_config
from utils.classes.StringBuilder import StringBuilder

def create_sample_data():
    """Create sample interaction data with various types of outliers."""
    np.random.seed(42)
    
    # Generate normal user interactions
    normal_users = []
    for user_id in range(1, 101):  # 100 normal users
        # Each user has 10-50 interactions over 30 days
        num_interactions = np.random.randint(10, 51)
        start_date = datetime.now() - timedelta(days=30)
        
        for i in range(num_interactions):
            # Random product (1-1000)
            product_id = np.random.randint(1, 1001)
            # Random action with realistic weights
            action = np.random.choice(['content_view', 'add_to_cart', 'add_to_wishlist', 'initiate_checkout', 'purchase'], 
                                   p=[0.6, 0.2, 0.1, 0.08, 0.02])
            # Random timestamp within last 30 days
            timestamp = start_date + timedelta(days=np.random.uniform(0, 30), 
                                            hours=np.random.uniform(0, 24),
                                            minutes=np.random.uniform(0, 60))
            
            normal_users.append({
                'user_uid': f'user_{user_id}',
                'product_id': product_id,
                'action': action,
                'created': int(timestamp.timestamp())
            })
    
    # Generate bot-like users (high frequency, regular patterns)
    bot_users = []
    for user_id in range(101, 111):  # 10 bot users
        # Bots have 200-500 interactions per day
        num_interactions = np.random.randint(200, 501)
        start_date = datetime.now() - timedelta(days=1)
        
        for i in range(num_interactions):
            product_id = np.random.randint(1, 1001)
            action = 'content_view'  # Bots mostly just view content
            # Very regular timing (every 10-30 seconds)
            timestamp = start_date + timedelta(seconds=i * np.random.uniform(10, 30))
            
            bot_users.append({
                'user_uid': f'bot_{user_id}',
                'product_id': product_id,
                'action': action,
                'created': int(timestamp.timestamp())
            })
    
    # Generate random clickers (low diversity, high repeat)
    random_clickers = []
    for user_id in range(111, 121):  # 10 random clickers
        # Click on same 5-10 products repeatedly
        favorite_products = np.random.choice(range(1, 1001), size=np.random.randint(5, 11), replace=False)
        num_interactions = np.random.randint(50, 201)
        start_date = datetime.now() - timedelta(days=7)
        
        for i in range(num_interactions):
            product_id = np.random.choice(favorite_products)
            action = np.random.choice(['content_view', 'add_to_cart'], p=[0.8, 0.2])
            timestamp = start_date + timedelta(days=np.random.uniform(0, 7),
                                            hours=np.random.uniform(0, 24),
                                            minutes=np.random.uniform(0, 60))
            
            random_clickers.append({
                'user_uid': f'clicker_{user_id}',
                'product_id': product_id,
                'action': action,
                'created': int(timestamp.timestamp())
            })
    
    # Combine all data
    all_interactions = normal_users + bot_users + random_clickers
    df = pd.DataFrame(all_interactions)
    
    return df

def demonstrate_outlier_detection():
    """Demonstrate different outlier detection methods."""
    print("=== OUTLIER DETECTION DEMONSTRATION ===\n")
    
    # Create sample data
    print("Creating sample data with various outlier types...")
    interactions = create_sample_data()
    print(f"Total interactions: {len(interactions):,}")
    print(f"Unique users: {interactions['user_uid'].nunique():,}")
    print(f"Unique products: {interactions['product_id'].nunique():,}\n")
    
    # 1. Default comprehensive outlier detection
    print("1. DEFAULT COMPREHENSIVE OUTLIER DETECTION")
    print("-" * 50)
    config = get_outlier_config("default")
    filtered_interactions, stats = comprehensive_outlier_detection(interactions, config)
    print(f"Original: {stats['original_count']:,} interactions")
    print(f"Final: {stats['final_count']:,} interactions")
    print(f"Removed: {stats['total_removed']:,} interactions ({stats['total_removed']/stats['original_count']*100:.1f}%)\n")
    
    # 2. Conservative outlier detection
    print("2. CONSERVATIVE OUTLIER DETECTION")
    print("-" * 50)
    config = get_outlier_config("conservative")
    filtered_interactions, stats = comprehensive_outlier_detection(interactions, config)
    print(f"Original: {stats['original_count']:,} interactions")
    print(f"Final: {stats['final_count']:,} interactions")
    print(f"Removed: {stats['total_removed']:,} interactions ({stats['total_removed']/stats['original_count']*100:.1f}%)\n")
    
    # 3. Aggressive outlier detection
    print("3. AGGRESSIVE OUTLIER DETECTION")
    print("-" * 50)
    config = get_outlier_config("aggressive")
    filtered_interactions, stats = comprehensive_outlier_detection(interactions, config)
    print(f"Original: {stats['original_count']:,} interactions")
    print(f"Final: {stats['final_count']:,} interactions")
    print(f"Removed: {stats['total_removed']:,} interactions ({stats['total_removed']/stats['original_count']*100:.1f}%)\n")
    
    # 4. Bot detection focused
    print("4. BOT DETECTION FOCUSED")
    print("-" * 50)
    config = get_outlier_config("bot_detection")
    filtered_interactions, stats = comprehensive_outlier_detection(interactions, config)
    print(f"Original: {stats['original_count']:,} interactions")
    print(f"Final: {stats['final_count']:,} interactions")
    print(f"Removed: {stats['total_removed']:,} interactions ({stats['total_removed']/stats['original_count']*100:.1f}%)\n")
    
    # 5. Custom configuration
    print("5. CUSTOM CONFIGURATION")
    print("-" * 50)
    custom_config = create_custom_config(
        max_user_interactions_per_day=100,
        z_score_threshold=2.5,
        min_diversity_ratio=0.15
    )
    filtered_interactions, stats = comprehensive_outlier_detection(interactions, custom_config)
    print(f"Original: {stats['original_count']:,} interactions")
    print(f"Final: {stats['final_count']:,} interactions")
    print(f"Removed: {stats['total_removed']:,} interactions ({stats['total_removed']/stats['original_count']*100:.1f}%)\n")

def demonstrate_individual_methods():
    """Demonstrate individual outlier detection methods."""
    print("=== INDIVIDUAL OUTLIER DETECTION METHODS ===\n")
    
    interactions = create_sample_data()
    config = get_outlier_config("default")
    
    # User-level outlier detection
    print("USER-LEVEL OUTLIER DETECTION")
    print("-" * 40)
    filtered_interactions, user_stats = detect_user_outliers(interactions, config)
    print(f"High frequency users: {user_stats['high_frequency_users']}")
    print(f"Bot-like users: {user_stats['bot_like_users']}")
    print(f"Low diversity users: {user_stats['low_diversity_users']}")
    print(f"Suspicious timing users: {user_stats['suspicious_timing_users']}")
    print(f"Total outlier users: {user_stats['total_outlier_users']}\n")
    
    # Session-level outlier detection
    print("SESSION-LEVEL OUTLIER DETECTION")
    print("-" * 40)
    filtered_interactions, session_stats = detect_session_outliers(interactions, config)
    print(f"Short sessions: {session_stats['short_sessions']}")
    print(f"High product sessions: {session_stats['high_product_sessions']}")
    print(f"Total outlier sessions: {session_stats['total_outlier_sessions']}\n")
    
    # Interaction-level outlier detection
    print("INTERACTION-LEVEL OUTLIER DETECTION")
    print("-" * 40)
    filtered_interactions, interaction_stats = detect_interaction_outliers(interactions, config)
    print(f"Z-score outliers: {interaction_stats['z_score_outliers']}")
    print(f"IQR outliers: {interaction_stats['iqr_outliers']}")
    print(f"Total outlier interactions: {interaction_stats['total_outlier_interactions']}\n")

def demonstrate_integration_with_als():
    """Demonstrate how to integrate outlier detection with ALS training."""
    print("=== INTEGRATION WITH ALS TRAINING ===\n")
    
    # This would be used in your actual ALS training pipeline
    print("Example usage in ALS training:")
    print("""
    # In your main training script:
    from utils.als import process_homepage_and_similar_products_recommendations
    from utils.outlier_config import get_outlier_config
    
    # Load your data
    raw_interactions = pd.read_csv('interactions.csv')
    raw_products = pd.read_csv('products.csv')
    
    # Train with outlier detection enabled (default: ecommerce config)
    process_homepage_and_similar_products_recommendations(
        raw_interactions, 
        raw_products, 
        enable_outlier_detection=True,
        outlier_config_name="ecommerce"
    )
    
    # Or with custom configuration
    process_homepage_and_similar_products_recommendations(
        raw_interactions, 
        raw_products, 
        enable_outlier_detection=True,
        outlier_config_name="bot_detection"
    )
    
    # Or disable outlier detection entirely
    process_homepage_and_similar_products_recommendations(
        raw_interactions, 
        raw_products, 
        enable_outlier_detection=False
    )
    """)

if __name__ == "__main__":
    # Run demonstrations
    demonstrate_outlier_detection()
    print("\n" + "="*60 + "\n")
    demonstrate_individual_methods()
    print("\n" + "="*60 + "\n")
    demonstrate_integration_with_als()