# uv add langchain-core langchain-google-vertexai google-genai python-dotenv 
from langchain_core.callbacks.base import BaseCallbackHandler
from typing import Any, Dict, List 
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

class DemoCallbackHandler(BaseCallbackHandler):
    """
    Custom callback handler for logging LangChain events.

    This handler logs the start and end of LLM, chain, and tool events, as well as errors and new tokens.
    Each method prints relevant information to the console for debugging and monitoring purposes.
    """

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        print(f"[LLM START] Model: {serialized.get('name', 'Unknown')} | Prompts: {prompts}")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        print(f"[NEW TOKEN] {token}", end="", flush=True)

    def on_llm_end(self, response, **kwargs: Any) -> None:
        print(f"[LLM END] Response: {response}")

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        print(f"[CHAIN START] {serialized.get('name', 'Unnamed Chain')} | Inputs: {inputs}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        print(f"[CHAIN END] Outputs: {outputs}")

    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        print(f"[CHAIN ERROR] {error}")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        print(f"[TOOL START] {serialized.get('name', 'Unnamed Tool')} | Input: {input_str}")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        print(f"[TOOL END] Output: {output}")

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        print(f"[TOOL ERROR] {error}")

# Example of custom callback handler 
if __name__ == "__main__":
    
    # Create LLM with custom callback
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", 
                                 callbacks=[DemoCallbackHandler()])

    # Run a simple query
    response = llm.invoke("Tell me a short joke about the Python Programming language.")
    print("\n\nFinal Output:", response)
