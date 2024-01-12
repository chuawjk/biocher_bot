import base64

from omegaconf import DictConfig

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage


class ImageDescriber:
    def __init__(self, cfg: DictConfig):
        """
        Initialise ImageDescriber class by preparing LLM.

        Args:
            cfg (DictConfig): ImageDescriber configurations.

        Returns:
            None
        """
        self.llm = ChatOpenAI(model=cfg.model_name, max_tokens=cfg.max_tokens)

    def describe(self, image_path):
        """
        Wrapper function to describe an image using LLM.

        Args:
            image_path (str): Path to image.

        Returns:
            image_desc (str): Description of image.
        """
        with open("prompts/image.txt", "r") as f:
            prompt = f.read()
        base64_image = self.encode_image(image_path)
        image_desc = self._describe(base64_image, prompt)
        return image_desc

    def encode_image(self, image_path):
        """
        Return base64 encoding of image.

        Args:
            image_path (str): Path to image.

        Returns:
            encoding (str): Base64 encoding of image.
        """
        with open(image_path, "rb") as image_file:
            encoding = base64.b64encode(image_file.read()).decode("utf-8")
            return encoding

    def _describe(self, img_base64, prompt):
        """
        Worker function to describe image.

        Args:
            img_base64 (str): Base64 encoding of image.
            prompt (str): Prompt to be fed to LLM.

        Returns:
            image_desc (str): Description of image.
        """
        msg = self.llm.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            },
                        },
                    ]
                )
            ]
        )
        image_desc = msg.content
        return image_desc
