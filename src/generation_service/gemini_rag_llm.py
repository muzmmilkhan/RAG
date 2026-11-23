import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class GeminiRAG:
    def __init__(self, model_name="gemini-4-argon", temperature=0.2):
        """
        Initializes the Gemini LLM model via the EURI OpenAI-compatible endpoint.
        Uses a vision-enabled Gemini model by default to handle image context (multimodal).
        """
        api_key = os.environ.get("EURI_API_KEY")
        if not api_key:
            raise ValueError("EURI_API_KEY environment variable not set.")

        base_url = os.environ.get("EURI_BASE_URL", "https://euron.one")

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )
        self.system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the provided image context (which represents pages of financial PDFs) to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Provide a detailed, accurate response based ONLY on the provided context."
        )

    def answer_question(self, question: str, base64_images: list[str]) -> str:
        """
        Answers a question using the provided list of base64 images as context.
        """
        print(f"Sending request to Gemini LLM ({self.llm.model_name}) ...")

        # Build the message content.
        content = [{"type": "text", "text": question}]

        # Append the base64 images
        for b64_img in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_img}"
                }
            })

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=content),
        ]

        response = self.llm.invoke(messages)
        return response.content
