from llama_index import download_loader
SimpleWebPageReader = download_loader("SimpleWebPageReader")

class WebpageData():
    def __init__(self, urls):
        self.urls = urls
        self.loader = SimpleWebPageReader()
        self.documents = self.loader.load_data(urls=self.urls)

