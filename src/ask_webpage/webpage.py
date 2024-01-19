from llama_index import download_loader
SimpleWebPageReader = download_loader("SimpleWebPageReader")

class WebpageData():
    def __init__(self, url):
        self.url = url
        self.loader = SimpleWebPageReader()
        self.document = self.loader.load_data(urls=[url])[0]

    def encode(self, embedder):
