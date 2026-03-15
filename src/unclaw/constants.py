"""Project-wide constants."""

APP_NAME = "unclaw"
DISPLAY_NAME = "Unclaw"

CONFIG_DIRECTORY_NAME = "config"
PROMPTS_DIRECTORY_NAME = "prompts"
DATA_DIRECTORY_NAME = "data"
LOGS_DIRECTORY_NAME = "logs"
SESSIONS_DIRECTORY_NAME = "sessions"
CACHE_DIRECTORY_NAME = "cache"
FILES_DIRECTORY_NAME = "files"

APP_CONFIG_FILE_NAME = "app.yaml"
MODELS_CONFIG_FILE_NAME = "models.yaml"
SYSTEM_PROMPT_FILE_NAME = "system.txt"
DATABASE_FILE_NAME = "app.db"
LOG_FILE_NAME = "runtime.log"

PROJECT_ROOT_ENV_VAR = "UNCLAW_PROJECT_ROOT"

RUNTIME_ERROR_REPLY = (
    "I could not complete that request locally right now. "
    "Check that Ollama is running and the selected model is available."
)
EMPTY_RESPONSE_REPLY = "The local model returned an empty response."
CHATBOT_MODE_FALLBACK_WARNING = (
    "Please note: the selected model profile does not support tools reliably. "
    "Unclaw will switch to Chatbot mode. "
    "Chatbot mode = simple conversation, no web research, no automation."
)
