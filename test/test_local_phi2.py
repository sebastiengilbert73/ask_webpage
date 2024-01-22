import logging
import sys
sys.path.append("../src/ask_webpage")
import local_llm

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')

def main():
    logging.debug("test_local_phi2.main()")

    llm = local_llm.phi2()
    question = "What is the mass of Jupiter?"
    response = llm.stream_complete(question)
    for word in response:
        print(word.delta, end="", flush=True)


if __name__ == '__main__':
    main()