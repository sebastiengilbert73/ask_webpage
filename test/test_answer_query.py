# Cf. https://docs.llamaindex.ai/en/stable/examples/low_level/response_synthesis.html
import logging
import test_retrieve_chunks
from llama_index.prompts import PromptTemplate

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')

def generate_response(retrieved_nodes, query_str, qa_prompt, llm):
    context_str = "\n\n".join([r.get_content() for r in retrieved_nodes])
    fmt_qa_prompt = qa_prompt.format(
        context_str=context_str, query_str=query_str
    )
    response = llm.complete(fmt_qa_prompt)
    return str(response), fmt_qa_prompt

def main():
    logging.info("test_answer_query.main()")

    nodes, llm, query = test_retrieve_chunks.main()

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


if __name__ == '__main__':
    main()