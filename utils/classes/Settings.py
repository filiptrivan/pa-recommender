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
    RECOMMENDATIONS_FILE_NAME = None
    APPLICATIONINSIGHTS_CONNECTION_STRING = None
    EMAIL_RECOMMENDATIONS_TEST = None

    def __init__(self):
        load_dotenv()
        self.ENV = os.getenv('ENV')
        self.API_KEY = os.getenv('API_KEY')
        self.EMAIL_SENDER = os.getenv('EMAIL_SENDER')
        self.EMAIL_SENDER_PASS = os.getenv('EMAIL_SENDER_PASS')
        self.EXCEPTION_EMAILS = os.getenv('EXCEPTION_EMAILS')
        self.AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        self.CONTAINER_NAME = os.getenv('CONTAINER_NAME')
        self.RECOMMENDATIONS_FILE_NAME = os.getenv('RECOMMENDATIONS_FILE_NAME')
        self.APPLICATIONINSIGHTS_CONNECTION_STRING = os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING')
        self.EMAIL_RECOMMENDATIONS_TEST = os.getenv('EMAIL_RECOMMENDATIONS_TEST')
