import os, json, dotenv
from typing import List
from pydantic import BaseModel
from groq import Groq


class NewsSummary(BaseModel):
    num_summaries: int
    headers: List[str]
    summaries: List[str]

class GroqClient():
    client: Groq

    def __init__(self, key=None):
        if key is None:
            print('-> Key not given, trying to retrieve from dotenv')
            found = dotenv.load_dotenv()
            if not found:
                raise ValueError('dotenv not found')

            api_key = os.getenv('GROQ_API_KEY')
            if api_key is None:
                raise ValueError('API key not found')

        self.client = Groq(api_key=key)

    def get_active_models(self) -> List[str]:
        """
        Returns available model names as json string
        """
        res = self.client.models.list().model_dump()
        if 'data' not in res:
            raise ValueError('cannot fetch models')

        models = res['data']
        active_models = [ model['id'] for model in models if model['active'] ]

        return active_models

    def get_summary(self, news_feed: str, model_id: str) -> NewsSummary:
        """
        @param news_feed: newline separate header and text containing all the news to be summaries
        @param model_id: which model to use to summarize the news

        @return model's summary of the news
        """
        schema_example = {
            "num_summaries": 3,
            "headers": [
                "Example Header 1",
                "Example Header 2",
                "Example Header 3"
                ],
            "summaries": [
                "Example summary 1.",
                "Example summary 2.",
                "Example summary 3."
                ]
            }


        chat_completion = self.client.chat.completions.create(
            messages = [
                {
                    'role': 'system',
                    'content': f'Sen, önemli haberleri özetleme konusunda uzmanlaşmış bir yapay zeka özetleyicisisin ve özetleri JSON formatında çıkartıyorsun.\nJSON, şu şemayı kullanmalıdır:\n{json.dumps(NewsSummary.model_json_schema(), indent=2)}\n\nÖrnek bir çıktı:\n{json.dumps(schema_example, indent=2)}',
                },
                {
                    'role': 'user',
                    'content': f'Aşağıdaki haber akışını özetle:\n{news_feed}'
                }
            ],
            model = model_id,
            temperature = 0,
            stream = False,
            response_format = { 'type': 'json_object' }
        )
        content = chat_completion.choices[0].message.content
        if content is None:
            raise ValueError('model failed to return a valid response')

        return NewsSummary.model_validate_json(content)
