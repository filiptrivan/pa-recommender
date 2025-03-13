class BusinessException(Exception):
    code = 400
    message: str

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
