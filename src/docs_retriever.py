import logging
from operator import itemgetter

import hydra
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.schema import format_document
from omegaconf import DictConfig


log = logging.getLogger(__name__)


class DocsRetriever:
    def __init__(self, cfg: DictConfig, vector_store):
        """
        Initialise DocsRetriever class by preparing vector store, LLM and chains.

        Args:
            cfg (DictConfig): DocsRetriever configurations.
            vector_store (langchain_community.vectorstores): Vector store to retrieve
                from.

        Returns:
            None
        """
        self.vector_store = vector_store
        self.num_to_retrieve = cfg.num_to_retrieve

        log.info("Preparing LLMs...")
        self.summary_llm = ChatOpenAI(model_name=cfg.model_name, temperature=0)
        self.question_llm = ChatOpenAI(model_name=cfg.model_name, temperature=0)

        log.info("Preparing prompts...")
        with open(cfg.summary_template_path, "r") as f:
            summary_template = f.read()
        self.summary_prompt = PromptTemplate.from_template(summary_template)
        with open(cfg.question_template_path, "r") as f:
            question_template = f.read()
        self.question_prompt = ChatPromptTemplate.from_template(question_template)
        self.document_prompt = PromptTemplate.from_template(template="{page_content}")

        log.info("Preparing retriever...")
        search_kwargs = {"k": cfg.num_to_retrieve}
        if cfg.search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = cfg.score_threshold
        if cfg.search_type == "mmr":
            search_kwargs["lambda_mult"] = cfg.lambda_mult

        self.retriever = self.vector_store.vector_store.as_retriever(
            search_type=cfg.search_type, search_kwargs=search_kwargs
        )

        log.info("Preparing chain...")
        inputs = RunnableParallel(
            standalone_question=RunnablePassthrough.assign(
                chat_history=lambda x: get_buffer_string(x["chat_history"])
            )
            | self.summary_prompt
            | self.summary_llm
            | StrOutputParser(),
        )
        context = {
            "context": itemgetter("standalone_question")
            | self.retriever
            | self.combine_documents,
            "question": lambda x: x["standalone_question"],
        }
        self.chat_rag_chain = (
            inputs
            | context
            | self.question_prompt
            | self.question_llm
        )

    def combine_documents(self, docs, document_separator="\n\n"):
        doc_strings = [format_document(doc, self.document_prompt) for doc in docs]
        return document_separator.join(doc_strings)


@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.1")
def run_standalone(cfg: DictConfig):
    from dotenv import load_dotenv
    from langchain_community.embeddings import OpenAIEmbeddings

    from src.vector_store import VectorStore

    _ = load_dotenv(cfg.credentials.dotenv_path)
    embedding_model = OpenAIEmbeddings()
    vector_store = VectorStore(
        vector_store_dir=cfg.vector_store.vector_store_dir,
        pdf_dir=cfg.vector_store.pdf_dir,
        chunk_size=cfg.vector_store.chunk_size,
        chunk_overlap=cfg.vector_store.chunk_overlap,
        embedding_model=embedding_model,
    )

    return run(cfg, vector_store)


def run(cfg: DictConfig, vector_store):
    retriever = DocsRetriever(cfg.docs_retriever, vector_store)

    chat_history = []
    while True:
        question = input("Enter question: ")
        answer = (
            retriever.chat_rag_chain.invoke(
                {"question": question, "chat_history": chat_history}
            )
        )
        print(answer.content)

        chat_history.append(HumanMessage(content=question))
        chat_history.append(answer)


if __name__ == "__main__":
    # python -m src.docs_retriever
    run_standalone()
