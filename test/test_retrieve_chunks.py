import logging
import test_embed_page

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')

def main():
    logging.info("test_retrieve_chunks.main()")

    vs_index, llm = test_embed_page.main()  # Returns a VectorSroreIndex

    ret = vs_index.as_retriever(similarity_top_k=5)
    nodes = ret.retrieve("How many moons does Saturn have?")
    logging.info(f"len(nodes) = {len(nodes)}")
    for node in nodes:
        print(node.text)
        print(f"len(node.text) = {len(node.text)}\n")

    return nodes, llm


if __name__ == '__main__':
    main()