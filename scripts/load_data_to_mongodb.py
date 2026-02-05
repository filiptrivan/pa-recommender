"""
Script for loading processed dataset into MongoDB.
Loads interactions and product recommendations data into MongoDB collections.
"""

import pandas as pd
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mongodb_client import get_mongodb_client


def load_interactions_data(file_path: str, batch_size: int = 10000):
    """
    Load interactions data from CSV into MongoDB.

    Args:
        file_path: Path to the interactions CSV file
        batch_size: Number of records to insert at once
    """
    print(f"Loading interactions data from: {file_path}")

    # Read CSV
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} interaction records")

    # Convert created timestamp to datetime
    df['created_datetime'] = pd.to_datetime(df['created'], unit='s')
    df['created_date'] = df['created_datetime'].dt.normalize()
    df['created_hour'] = df['created_datetime'].dt.hour
    df['created_day_of_week'] = df['created_datetime'].dt.dayofweek
    df['created_month'] = df['created_datetime'].dt.month

    # Convert to records
    records = df.to_dict('records')

    # Get MongoDB collection
    client = get_mongodb_client()
    collection = client.get_collection('interactions')

    # Drop existing collection
    print("Dropping existing 'interactions' collection...")
    client.drop_collection('interactions')

    # Insert data in batches
    print(f"Inserting {len(records)} records in batches of {batch_size}...")
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        collection.insert_many(batch)
        print(f"Inserted batch {i // batch_size + 1}: {i + len(batch)}/{len(records)} records")

    # Create indexes for better query performance
    print("Creating indexes...")
    collection.create_index('user_uid')
    collection.create_index('product_id')
    collection.create_index('action')
    collection.create_index('created')
    collection.create_index([('user_uid', 1), ('product_id', 1)])
    collection.create_index([('user_uid', 1), ('action', 1)])
    collection.create_index('created_date')

    print(f"✓ Successfully loaded {len(records)} interaction records into MongoDB")
    return len(records)


def load_product_recommendations_data(file_path: str, batch_size: int = 10000):
    """
    Load product recommendations data from CSV into MongoDB.

    Args:
        file_path: Path to the product_product CSV file
        batch_size: Number of records to insert at once
    """
    print(f"\nLoading product recommendations data from: {file_path}")

    # Read CSV
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} recommendation records")

    # Convert to records
    records = df.to_dict('records')

    # Get MongoDB collection
    client = get_mongodb_client()
    collection = client.get_collection('product_recommendations')

    # Drop existing collection
    print("Dropping existing 'product_recommendations' collection...")
    client.drop_collection('product_recommendations')

    # Insert data in batches
    print(f"Inserting {len(records)} records in batches of {batch_size}...")
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        collection.insert_many(batch)
        print(f"Inserted batch {i // batch_size + 1}: {i + len(batch)}/{len(records)} records")

    # Create indexes
    print("Creating indexes...")
    collection.create_index('product_to_recommend_id')
    collection.create_index('product_for_recommendation_id')
    collection.create_index('interaction_weight')
    collection.create_index([('product_to_recommend_id', 1), ('interaction_weight', -1)])

    print(f"✓ Successfully loaded {len(records)} recommendation records into MongoDB")
    return len(records)


def verify_data_load():
    """Verify that data has been loaded correctly."""
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)

    client = get_mongodb_client()

    # Verify interactions
    interactions = client.get_collection('interactions')
    interaction_count = interactions.count_documents({})
    print(f"\nInteractions collection: {interaction_count} documents")

    # Sample interaction
    sample_interaction = interactions.find_one()
    print("\nSample interaction:")
    for key, value in sample_interaction.items():
        if key != '_id':
            print(f"  {key}: {value}")

    # Verify product recommendations
    recommendations = client.get_collection('product_recommendations')
    recommendation_count = recommendations.count_documents({})
    print(f"\nProduct Recommendations collection: {recommendation_count} documents")

    # Sample recommendation
    sample_recommendation = recommendations.find_one()
    print("\nSample product recommendation:")
    for key, value in sample_recommendation.items():
        if key != '_id':
            print(f"  {key}: {value}")

    # List all indexes
    print("\nIndexes on 'interactions' collection:")
    for index in interactions.list_indexes():
        print(f"  - {index['name']}: {index.get('key', {})}")

    print("\nIndexes on 'product_recommendations' collection:")
    for index in recommendations.list_indexes():
        print(f"  - {index['name']}: {index.get('key', {})}")

    print("\n" + "="*60)


def main():
    """Main function to orchestrate data loading."""
    print("="*60)
    print("PA RECOMMENDER - MongoDB Data Loader")
    print("="*60)

    # Test MongoDB connection
    client = get_mongodb_client()
    if not client.ping():
        print("ERROR: Cannot connect to MongoDB!")
        print("Make sure MongoDB is running: docker-compose up -d")
        return

    print("✓ MongoDB connection successful\n")

    # Define file paths
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    interactions_file = os.path.join(base_path, 'data', 'interactions.csv')
    recommendations_file = os.path.join(base_path, 'data', 'product_product.csv')

    # Check if files exist
    if not os.path.exists(interactions_file):
        print(f"ERROR: Interactions file not found: {interactions_file}")
        return

    if not os.path.exists(recommendations_file):
        print(f"ERROR: Recommendations file not found: {recommendations_file}")
        return

    # Load data
    try:
        load_interactions_data(interactions_file)
        load_product_recommendations_data(recommendations_file)
        verify_data_load()

        print("\n" + "="*60)
        print("✓ DATA LOADING COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nYou can now:")
        print("  1. Access MongoDB at: mongodb://localhost:27017")
        print("  2. View data in Mongo Express at: http://localhost:8081")
        print("     Username: admin")
        print("     Password: admin123")
        print("  3. Run analytical queries from Python notebooks")

    except Exception as e:
        print(f"\nERROR: Failed to load data: {e}")
        import traceback
        traceback.print_exc()

    finally:
        client.close()


if __name__ == "__main__":
    main()
