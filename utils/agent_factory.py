import os
from typing import Union
from utils.file_management import load_json
from core.errors import ModelNotSupportedError
from ratbench.agents import AzureChatGPTAgent, BedrockAgent, VertexAgent


def agent_factory(
        agent_name: str,
        model: str,
        region: str = "us-east-1"
) -> Union[AzureChatGPTAgent, BedrockAgent, VertexAgent]:
    """Returns an agent given its name and the model"""
    if model.startswith("gpt"):
        return AzureChatGPTAgent(
            agent_name=agent_name,
            model=model,
            azure_endpoint=os.getenv("OPENAI_API_BASE_2"),
            api_key=os.getenv("OPENAI_API_KEY_2"),
            api_version=os.getenv("OPENAI_API_VERSION_2")
        )
    elif (model.startswith("us.anthropic") or
          model.startswith("meta") or
          model.startswith("us.meta") or
          model.startswith("anthropic") or
          model.startswith("us.deepseek")):
        return BedrockAgent(
                agent_name=agent_name,
                model=model,
                region=region
            )
    elif model.startswith("gemini"):
        google_json_creds = load_json(
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        )
        return VertexAgent(
                agent_name=agent_name,
                model=model,
                google_json_creds=google_json_creds
            )
    else:
        raise ModelNotSupportedError(f"Model {model} is not supported")
