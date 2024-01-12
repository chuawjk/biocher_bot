import logging
import os

from dotenv import load_dotenv
import hydra
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from omegaconf import DictConfig, OmegaConf
from telegram import Update
from telegram.ext import (
    filters,
    MessageHandler,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

from src.docs_retriever import DocsRetriever
from src.image_describer import ImageDescriber
from src.vector_store import VectorStore

log = logging.getLogger(__name__)

cfg = OmegaConf.load("conf/config.yaml")

_ = load_dotenv(cfg.credentials.dotenv_path)


# Global dictionary to store chat history for each chat session
# TODO: make this more scalable
chat_histories = {}

log.info("Loading vector store...")
embedding_model = OpenAIEmbeddings()
vector_store = VectorStore(cfg.vector_store, embedding_model)

log.info("Loading image describer...")
image_describer = ImageDescriber(cfg.image_describer)

# TODO: change to cosine similarity
log.info("Loading retriever...")
retriever = DocsRetriever(cfg.docs_retriever, vector_store)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles the /start command.

    Args:
        update (Update): The update object from Telegram.
        context (ContextTypes.DEFAULT_TYPE): The context object from Telegram.

    Returns
        None
    """
    log.info("Received start message")
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="🧬 Hi, I'm Biocher Bot! Ask me anything related to A Level Biology!",
    )


async def update_chat_history(chat_id: int, message):
    """
    Updates the chat history for a given chat session.

    Args:
        chat_id (int): Unique chat ID.
        message (langchain_core.messages.HumanMessage): The message to be added
            to the chat history.

    Returns:
        None
    """
    if chat_id not in chat_histories:
        chat_histories[chat_id] = []
    chat_histories[chat_id].append(message)


async def question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles the user's question by invoking RAG chain.

    Args:
        update (Update): The update object from Telegram.
        context (ContextTypes.DEFAULT_TYPE): The context object from Telegram.

    Returns:
        None
    """
    chat_id = update.effective_chat.id

    if update.message.photo:
        log.info("Received image")
        # Get the highest quality photo
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        image_path = f"downloads/{photo.file_id}.jpg"
        await file.download_to_drive(image_path)

        # Update chat history with the image description
        # TODO: run this in parallel with retriever
        log.info("Describing image...")
        image_desc = image_describer.describe(image_path)

        # Set image caption as user question
        user_question = image_desc + "\n\n" + update.message.caption
        log.debug(f"User question: {user_question}")

    else:
        log.info(f"Received question message")
        user_question = update.message.text
        image_desc = None

    await update_chat_history(chat_id, HumanMessage(content=user_question))

    # Retrieve bot's response using the updated chat history
    log.info("Retrieving bot's response...")
    bot_answer = retriever.chat_rag_chain.invoke(
        {
            "question": user_question,
            "chat_history": chat_histories.get(chat_id),
            "image_desc": image_desc,
        }
    )
    await update_chat_history(chat_id, bot_answer)

    # Send the bot's response to the user
    await context.bot.send_message(
        chat_id=chat_id,
        text=bot_answer.content,
    )


@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.1")
def run(cfg: DictConfig):
    _ = load_dotenv(cfg.credentials.dotenv_path)

    log.info("Starting bot...")
    token = os.getenv("TELEGRAM_TOKEN")
    application = ApplicationBuilder().token(token).build()

    start_handler = CommandHandler("start", start)
    question_handler = MessageHandler(
        filters.PHOTO | filters.TEXT & (~filters.COMMAND), question
    )

    application.add_handler(start_handler)
    application.add_handler(question_handler)

    application.run_polling()


if __name__ == "__main__":
    # python -m src.bot
    run()
