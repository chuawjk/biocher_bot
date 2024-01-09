import logging
from pathlib import Path
import shutil

import hydra
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from omegaconf import DictConfig
import openai

log = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        vector_store_dir: str,
        pdf_dir: str,
        chunk_size: int,
        chunk_overlap: int,
        embedding_model,
    ):
        """
        Initialise VectorStore class by preparing configs.

        Args:
            vector_store_dir (str): Path to existing vector store or where to save 
                new one.
            pdf_dir (str): Path to directory containing PDFs.
            chunk_size (int): Number of characters per chunk after splitting PDF.
            chunk_overlap (int): Number of characters to overlap between chunks.
            embedding_model (langchain_community.embeddings): Model to convert 
                chunks into vectors.

        Returns:
            None
        """
        self.vector_store_dir = vector_store_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model

        self.pdf_dir = pdf_dir
        self.pdf_paths = list(Path(self.pdf_dir).glob("*.pdf")) + list(
            Path(self.pdf_dir).glob("*.PDF")
        )

        self.vector_store = None
        self.prepare()

    def prepare(self):
        """
        Prepare vector store by loading from disk and updating it if any new PDFs
        are found. If there is no existing vector store, a new one is created from
        provided PDF paths.

        Returns:
            None
        """
        if not Path(self.vector_store_dir).exists():
            log.info("No existing vector store found.")
            log.info("Creating new vector store...")
            Path(self.vector_store_dir).mkdir(parents=True)
            try:
                self.add_docs(self.pdf_paths, self.embedding_model)
            except (
                openai.APIConnectionError,
                openai.AuthenticationError,
                openai.BadRequestError,
            ) as e:
                log.error(e)
                log.error("Vector store creation failed. Exiting...")
                shutil.rmtree(self.vector_store_dir)
                exit(1)

        else:
            log.info("Loading existing vector store...")
            self.vector_store = FAISS.load_local(
                self.vector_store_dir, self.embedding_model
            )

            # Compare indexed documents with pdf_paths
            with open(Path(self.vector_store_dir) / "indexed_documents.txt", "r") as f:
                indexed_documents = f.read().split("\n")
            new_documents = [
                p.name for p in self.pdf_paths if str(p.name) not in indexed_documents
            ]

            # If there are any new documents, add them to the vector store
            if len(new_documents) > 0:
                log.info(f"Found {len(new_documents)} new documents to index.")
                self.new_pdf_paths = [
                    Path(self.pdf_dir) / filename for filename in new_documents
                ]
                self.add_docs(self.new_pdf_paths, self.embedding_model)

            else:
                log.info("No new documents to index.")

    def add_docs(self, pdf_paths, embedding_model):
        """
        Add documents to vector store. If there is no existing vector store, a new one
        is created from provided PDF paths. Else, the new documents are added to the
        existing vector store.

        Args:
            pdf_paths (list): List of pathlib.Path objects containing PDF paths.
            embedding_model (langchain.embeddings): Embedding model to use.

        Returns:
            None
        """
        for pdf_path in pdf_paths:
            log.info(f"Loading {pdf_path.name}...")
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()

            log.info(f"Splitting into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            split_docs = text_splitter.split_documents(documents)

            log.info(f"Embedding and adding to vector store. This may take a while...")
            cur_vector_store = FAISS.from_documents(split_docs, embedding_model)

            if self.vector_store is None:
                self.vector_store = cur_vector_store
            else:
                self.vector_store.merge_from(cur_vector_store)

            self.vector_store.save_local(self.vector_store_dir)

            with open(Path(self.vector_store_dir) / "indexed_documents.txt", "a") as f:
                f.write(f"{Path(pdf_path).name}\n")

        log.info("Done adding documents to vector store.")


@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.1")
def run_standalone(cfg: DictConfig):
    from dotenv import load_dotenv
    from langchain_community.embeddings import OpenAIEmbeddings

    _ = load_dotenv(cfg.credentials.dotenv_path)

    embedding_model = OpenAIEmbeddings()

    return run(cfg, embedding_model)


def run(cfg: DictConfig, embedding_model):
    return VectorStore(
        vector_store_dir=cfg.vector_store.vector_store_dir,
        pdf_dir=cfg.vector_store.pdf_dir,
        chunk_size=cfg.vector_store.chunk_size,
        chunk_overlap=cfg.vector_store.chunk_overlap,
        embedding_model=embedding_model,
    )


if __name__ == "__main__":
    # python -m src.vector_store
    run_standalone()
