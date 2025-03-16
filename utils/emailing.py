import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class Emailing:
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.sender_email = os.getenv('EMAIL_SENDER')
        self.sender_password = os.getenv('EMAIL_SENDER_PASS')

    def send_email(self, recipients: str, message: str, subject: str):
        if self.sender_email is None:
            return
        
        if self.sender_password is None:
            return
        
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = recipients
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))
        
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.sendmail(self.sender_email, recipients, msg.as_string())