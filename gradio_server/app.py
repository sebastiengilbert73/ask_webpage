import gradio as gr
import sys
sys.path.append("../src/ask_webpage")
import local_llm
import webpage
import local_embedder
import platform
if not 'windows' in platform.system().lower():
    __import__('pysqlite3')  # https://gist.github.com/defulmere/8b9695e415a44271061cc8e272f3c300
    # Requires pip install pysqlite3-binary
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from llama_index import VectorStoreIndex, ServiceContext, StorageContext
from llama_index.vector_stores import ChromaVectorStore
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')

llm = local_llm.zephyr_7b_alpha()
embedder = local_embedder.bge_small_en_v1p5()
service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embedder,
        chunk_size=512,
        chunk_overlap=128
    )


def search_website(website_url, query):
    webpage_extractor = webpage.WebpageData([website_url])
    documents = webpage_extractor.documents

    db_name = "webpage"
    db = chromadb.Client()
    chroma_collection = db.create_collection(db_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vs_index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context,
        storage_context=storage_context,
        show_progress=True
    )

    ret = vs_index.as_retriever(similarity_top_k=5)

    nodes = ret.retrieve(query)
    logging.info(f"len(nodes) = {len(nodes)}")
    for node in nodes:
        logging(f"{node.text}\n\n ++++++++++++++++")

    qa_prompt = PromptTemplate(
        """\
        Context information is below.
        ---------------------
        {context_str}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        Query: {query_str}
        Answer: \
        """
    )

    response, fmt_qa_prompt = generate_response(nodes, query, qa_prompt, llm)
    logging.info(f"***** Response: *****\n{response}\n\n")
    logging.info(f"***** Formatted Prompt: *****\n{fmt_qa_prompt}\n\n")

    return response

demo = gr.Interface(
    fn=search_website,
    inputs=["text", "test"],
    outputs=["text"]
)

demo.launch(server_name="0.0.0.0", server_port=7860)