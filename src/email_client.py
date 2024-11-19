import smtplib, ssl, dotenv, os
from httpx import _content
from jinja2 import Environment, FileSystemLoader
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from llm import NewsSummary

class EmailClient():
    PORT = smtplib.SMTP_SSL_PORT
    SMTP_SERVER = 'smtp.gmail.com'
    SENDER_EMAIL = 'turkishnewsfeeder@gmail.com'
    RECEIVER_EMAILS = [
        'dmrarikan@gmail.com',
        'atamertiel@gmail.com',
    ]

    def __init__(self, secret=None):

        # get secret
        if secret is None:
            print('-> Key not given, trying to retrieve from dotenv')
            found = dotenv.load_dotenv()
            if not found:
                raise ValueError('dotenv not found')

            secret = os.getenv('GMAIL_APP_PASSWORD')
            if secret is None:
                raise ValueError('email app-key not found')

        self.secret = secret

        # setup html template
        loader = FileSystemLoader(searchpath='./src')
        env = Environment(loader=loader)
        self.html_template = env.get_template('template.html')

    def send_email_to_all(self, summary: NewsSummary) -> None:
        """
        Sends email (summary content of the news) to all subscribed users
        """
        html_content = self.html_template.render(summary=summary)


        # send email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(self.SMTP_SERVER, self.PORT, context=context) as server:
            server.login(self.SENDER_EMAIL, self.secret)
            for receiver_email in self.RECEIVER_EMAILS:
                message = self._construct_message(subject='Yeni haberler', sender_email=self.SENDER_EMAIL, html_content=html_content)
                message['To'] = receiver_email
                try:
                    server.sendmail(self.SENDER_EMAIL, receiver_email, message.as_string())
                except Exception as e:
                    print(f'Failed to send email to {receiver_email}: {e}')

    def _construct_message(self, subject: str, sender_email: str, html_content: str) -> MIMEMultipart:
        message = MIMEMultipart('alternative')
        message['Subject'] = subject
        message['From'] = sender_email
        part = MIMEText(html_content, 'html')
        message.attach(part)
        return message


if __name__ == '__main__':
    # Data to be sent
    data = [
        {"header": "HEADER1", "content": "CONTENT1"},
        {"header": "HEADER2", "content": "CONTENT2"},
        {"header": "HEADER3", "content": "CONTENT3"},
    ]

    summary = NewsSummary(num_summaries=3, headers=['HEADER1', 'HEADER2', 'HEADER3'], summaries=['SUMMARY1', 'SUMMARY2', 'SUMMAR3'])

    email_client = EmailClient()
    email_client.send_email_to_all(summary)
