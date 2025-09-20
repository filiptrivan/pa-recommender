import os
from dotenv import load_dotenv
from utils.classes.Singleton import Singleton

class Settings(metaclass=Singleton):
    ENV = None
    API_KEY = None
    EMAIL_SENDER = None
    EMAIL_SENDER_PASS = None
    EXCEPTION_EMAILS = None
    AZURE_STORAGE_CONNECTION_STRING = None
    CONTAINER_NAME = None
    APPLICATIONINSIGHTS_CONNECTION_STRING = None
    TEST_EMAIL_FOR_RECOMMENDATIONS = None
    TEST_PRODUCT_FOR_SIMILAR_PRODUCTS = None
    TEST_PRODUCT_FOR_CROSS_SELL = None
    REDIS_USERNAME = None
    REDIS_PASSWORD = None
    API_URL = None
    BEARER_TOKEN = None

    def __init__(self):
        load_dotenv()
        self.ENV = os.getenv('ENV')
        self.API_KEY = os.getenv('API_KEY')
        self.EMAIL_SENDER = os.getenv('EMAIL_SENDER')
        self.EMAIL_SENDER_PASS = os.getenv('EMAIL_SENDER_PASS')
        self.EXCEPTION_EMAILS = os.getenv('EXCEPTION_EMAILS')
        self.AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        self.CONTAINER_NAME = os.getenv('CONTAINER_NAME')
        self.APPLICATIONINSIGHTS_CONNECTION_STRING = os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING')
        self.TEST_EMAIL_FOR_RECOMMENDATIONS = os.getenv('TEST_EMAIL_FOR_RECOMMENDATIONS')
        self.TEST_PRODUCT_FOR_SIMILAR_PRODUCTS = os.getenv('TEST_PRODUCT_FOR_SIMILAR_PRODUCTS')
        self.TEST_PRODUCT_FOR_CROSS_SELL = os.getenv('TEST_PRODUCT_FOR_CROSS_SELL')
        self.REDIS_USERNAME = os.getenv('REDIS_USERNAME')
        self.REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')
        self.API_URL = os.getenv('API_URL')
        self.BEARER_TOKEN = os.getenv('BEARER_TOKEN')
