import os, json, dotenv
from typing import List
from groq import Groq
from scraper import Feed

from models import NewsSummary, HeaderIndices


class GroqClient():
    client: Groq
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
    header_indices_example = {
        "indices": [
            4, 1, 10, 3
        ]
    }

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
        chat_completion = self.client.chat.completions.create(
            messages = [
                {
                    'role': 'system',
                    'content': f'Sen, önemli haberleri özetleme konusunda uzmanlaşmış bir yapay zeka özetleyicisisin ve özetleri JSON formatında çıkartıyorsun.\nJSON, şu şemayı kullanmalıdır:\n{json.dumps(NewsSummary.model_json_schema(), indent=2)}\n\nÖrnek bir çıktı:\n{json.dumps(self.schema_example, indent=2)}',
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

    def summarize_feed(self, feed: Feed, model_id: str) -> NewsSummary:
        """
        @param feed: Feed object containing website url and entries of the news
        @param model_id: which model to use to summarize the news

        @return model's summary of the news as NewsSummary object
        """

        # concat the headers and contents with newline and each entry (header,content) is separated
        # with '\n\n'
        news_feed = [ f'{entry.header}\n{entry.content}' for entry in feed.entries ]
        news_feed = '\n\n'.join(news_feed)

        chat_completion = self.client.chat.completions.create(
            messages = [
                {
                    'role': 'system',
                    'content': f'Sen, önemli haberleri özetleme konusunda uzmanlaşmış bir yapay zeka özetleyicisisin ve özetleri JSON formatında çıkartıyorsun.\nJSON, şu şemayı kullanmalıdır:\n{json.dumps(NewsSummary.model_json_schema(), indent=2)}\n\nÖrnek bir çıktı:\n{json.dumps(self.schema_example, indent=2)}',
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

    def choose_headers(self, feed: Feed, count: int, model_id: str) -> list[int]:
        """
        Let the model choose the most important news sorted by importance

        @param feed: news feed containing all the headers and content
        @param count: number of header indice to return, must be less than the given num. of entries
        @param model_id: which model to use

        @return indices of the most important headers [usize; count]
        """
        assert len(feed.entries) >= count, f'Cannot return {count} number of headers from {len(feed.entries)} number of available options'

        headers = ''
        current_idx = 0
        for entry in feed.entries:
            headers += f'{current_idx} {entry.header}\n'
            current_idx += 1

        last_index = current_idx

        chat_completion = self.client.chat.completions.create(
            messages = [
                {
                    'role': 'system',
                    'content': f'Sen, önemli haber başlıklarını seçme konusunda uzmanlaşmış bir yapay zeka modelisin ve en önemli haber başlıklarının index\'lerini JSON formatında çıkartıyorsun (index 0 ile başlar).\nJSON, şu şemayı kullanmalıdır:\n{json.dumps(HeaderIndices.model_json_schema(), indent=2)}\n\nÖrnek bir çıktı:\n{json.dumps(self.header_indices_example, indent=2)}',
                },
                {
                    'role': 'user',
                    'content': f'Aşağıdan önem sırasına göre en önemli {count} haberi seç:\n{headers}'
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

        headers = HeaderIndices.model_validate_json(content)

        for idx in headers.indices:
            if idx < 0 or idx > last_index:
                raise ValueError('model returned invalid header index')

        return headers.indices

    def get_num_tokens(self, content: str) -> int:
        # FIXME: this is just an estimate. learn which tokenizer are being used by the preferred models
        #        to calculate the encodings and then num. tokens given string
        return len(content) // 4

if __name__ == '__main__':
    client = GroqClient()
    client.get_active_models


