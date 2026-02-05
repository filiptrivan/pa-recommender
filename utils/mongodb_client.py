"""
MongoDB client utility module for PA Recommender project.
Provides connection management and common database operations.
"""

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MongoDBClient:
    """Singleton MongoDB client for managing database connections."""

    _instance: Optional['MongoDBClient'] = None
    _client: Optional[MongoClient] = None
    _database: Optional[Database] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDBClient, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize MongoDB client with connection parameters."""
        if self._client is None:
            self.host = os.getenv('MONGO_HOST', 'localhost')
            self.port = int(os.getenv('MONGO_PORT', 27017))
            self.username = os.getenv('MONGO_USERNAME', 'admin')
            self.password = os.getenv('MONGO_PASSWORD', 'admin123')
            self.database_name = os.getenv('MONGO_DATABASE', 'pa_recommender')
            self.auth_source = os.getenv('MONGO_AUTH_SOURCE', 'admin')

    def connect(self) -> Database:
        """
        Establish connection to MongoDB and return database instance.

        Returns:
            Database: MongoDB database instance
        """
        if self._client is None:
            connection_string = (
                f"mongodb://{self.username}:{self.password}@"
                f"{self.host}:{self.port}/?authSource={self.auth_source}"
            )
            self._client = MongoClient(connection_string)
            self._database = self._client[self.database_name]
            print(f"Connected to MongoDB: {self.database_name}")

        return self._database

    def get_database(self) -> Database:
        """
        Get current database instance, connecting if necessary.

        Returns:
            Database: MongoDB database instance
        """
        if self._database is None:
            return self.connect()
        return self._database

    def get_collection(self, collection_name: str) -> Collection:
        """
        Get a specific collection from the database.

        Args:
            collection_name: Name of the collection

        Returns:
            Collection: MongoDB collection instance
        """
        db = self.get_database()
        return db[collection_name]

    def close(self):
        """Close MongoDB connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._database = None
            print("MongoDB connection closed")

    def ping(self) -> bool:
        """
        Test MongoDB connection.

        Returns:
            bool: True if connection is successful
        """
        try:
            db = self.get_database()
            db.command('ping')
            return True
        except Exception as e:
            print(f"MongoDB ping failed: {e}")
            return False

    def list_collections(self) -> list:
        """
        List all collections in the current database.

        Returns:
            list: List of collection names
        """
        db = self.get_database()
        return db.list_collection_names()

    def drop_collection(self, collection_name: str):
        """
        Drop a collection from the database.

        Args:
            collection_name: Name of the collection to drop
        """
        db = self.get_database()
        db.drop_collection(collection_name)
        print(f"Collection '{collection_name}' dropped")


def get_mongodb_client() -> MongoDBClient:
    """
    Factory function to get MongoDB client instance.

    Returns:
        MongoDBClient: Singleton MongoDB client instance
    """
    return MongoDBClient()
