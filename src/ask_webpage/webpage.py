from llama_index import download_loader
UnstructuredURLLoader = download_loader("UnstructuredURLLoader")

class WebpageData():
    def __init__(self, url):
        self.url = url
        self.loader = UnstructuredURLLoader(urls=[url], continue_on_failure=False)
        self.loader.load()

