import logging
import test_embed_page

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')

def main():
    logging.info("test_retrieve_chunks.main()")

    vs_index = test_embed_page.main()  # Returns a VectorSroreIndex

    ret = vs_index.as_retriever(similarity_top_k=5)
    query_engine = vs_index.as_query_engine(similarity_top_k=5)


if __name__ == '__main__':
    main()