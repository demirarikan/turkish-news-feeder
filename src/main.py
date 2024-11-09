import dotenv
import os

dotenv.load_dotenv()
key = os.getenv('GROQ_API_KEY')
print(key)

# TODO: cronjob should run this main.py


# TODO: use scraper to scrape rss feeds from atv, and other news feed

# TODO: prepare groq requets to send the feed and get summary (prepare prompt)

# TODO: retrieve prompts from groq and embed them into an email template

# TODO: send the email to users subscribed?


