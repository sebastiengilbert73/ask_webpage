import logging
__import__('pysqlite3')  # https://gist.github.com/defulmere/8b9695e415a44271061cc8e272f3c300
# Requires pip install pysqlite3-binary
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb

from llama_index import VectorStoreIndex, ServiceContext, StorageContext
from llama_index.vector_stores import ChromaVectorStore

sys.path.append("../src/ask_webpage")
import local_llm
import local_embedder
import webpage

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')

def main():
    logging.info("test_embed_page.main()")

    llm = local_llm.zephyr_7b_alpha()
    embedder = local_embedder.bge_small_en_v1p5()

    db_directory = "./chroma_db"
    db_name = "planets"
    db = chromadb.PersistentClient(path=db_directory)
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embedder,
        chunk_size=512,
        chunk_overlap=128
    )

    if db_name not in [c.name for c in db.list_collections()]:  # Create the index
        logging.info(f"Building the index {db_name}...")
        planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
        urls = [f"https://en.wikipedia.org/wiki/{planet}_(planet)" for planet in planets]
        webpage_extractor = webpage.WebpageData(urls)
        documents = webpage_extractor.documents

        chroma_collection = db.create_collection(db_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vs_index = VectorStoreIndex.from_documents(
            documents,
            service_context=service_context,
            storage_context=storage_context,
            show_progress=True
        )
    else:  # The index was already created
        logging.info(f"Loading the {db_name} from {db_directory}...")
        chroma_collection = db.get_collection(db_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        vs_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            service_context=service_context
        )


if __name__ == '__main__':
    main()