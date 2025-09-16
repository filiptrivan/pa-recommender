import logging
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders

from utils.classes.Settings import Settings

logger = logging.getLogger(__name__)

class Emailing:
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.sender_email = Settings().EMAIL_SENDER
        self.sender_password = Settings().EMAIL_SENDER_PASS

    def send_email(self, recipients: str, subject: str, message: str):
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
            server.set_debuglevel(1)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.sendmail(self.sender_email, recipients, msg.as_string())

    def send_email_with_attachments(self, recipients: str, subject: str, message: str, attachments: list[tuple[str, bytes, str]] | None = None, html: bool = False):
        if self.sender_email is None:
            return

        if self.sender_password is None:
            return

        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = recipients
        msg['Subject'] = subject

        subtype = 'html' if html else 'plain'
        msg.attach(MIMEText(message, subtype))

        if attachments:
            for filename, content_bytes, mime_type in attachments:
                try:
                    if mime_type.startswith('image/'):
                        part = MIMEImage(content_bytes, _subtype=mime_type.split('/', 1)[1])
                    else:
                        part = MIMEBase(*mime_type.split('/', 1))
                        part.set_payload(content_bytes)
                        encoders.encode_base64(part)
                    part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
                    msg.attach(part)
                except Exception:
                    logger.exception("Failed attaching file to email")

        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.set_debuglevel(1)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.sendmail(self.sender_email, recipients, msg.as_string())

    def send_email_and_log_info(self, subject: str, message: str, attachments: list[tuple[str, bytes, str]] | None = None, html: bool = False):
        if attachments:
            self.send_email_with_attachments(Settings().EXCEPTION_EMAILS, subject, message, attachments=attachments, html=html)
        else:
            self.send_email(Settings().EXCEPTION_EMAILS, subject, message)
        logger.info(f'{subject}\n{message}')