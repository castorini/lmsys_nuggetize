import os
from typing import Dict, Optional

from dotenv import load_dotenv


def get_azure_openai_args() -> Dict[str, Optional[str]]:
    load_dotenv(dotenv_path=".env")
    azure_args = {
        "api_type": "azure",
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "api_base": os.getenv("AZURE_OPENAI_API_BASE"),
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    }

    # Sanity check
    assert all(
        list(azure_args.values())
    ), "Ensure that `AZURE_OPENAI_API_BASE`, `AZURE_OPENAI_API_VERSION` are set"
    for key, value in azure_args.items():
        if value is None:
            raise ValueError(f"{key} not found in environment variables")
    else:
        return azure_args


azure_args = get_azure_openai_args()
api_type = "azure"
api_base = azure_args.get("api_base")
api_version = azure_args.get("api_version")
api_keys = azure_args.get("api_key")
