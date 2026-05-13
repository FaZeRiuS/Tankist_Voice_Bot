import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

def setup_gemini(api_key: str):
    if not api_key:
        logger.warning("GEMINI_API_KEY is not set. LLM features will be disabled.")
        return False
    genai.configure(api_key=api_key)
    return True

async def _get_model(name: str = "gemini-flash-latest"):
    """Internal helper to get a model or fallback to gemini-pro-latest."""
    try:
        return genai.GenerativeModel(name)
    except Exception:
        logger.warning(f"Model {name} not found, falling back to gemini-pro-latest")
        return genai.GenerativeModel("gemini-pro-latest")

async def summarize_user_history(history: list[str]) -> str | None:
    """Analyze user history and return a 1-sentence personality summary."""
    if not history:
        return None
    
    text_history = "\n".join(history)
    prompt = (
        "Based on the following chat messages from a user, summarize their interests, "
        "personality, and common topics they talk about. "
        "Keep it extremely concise (one sentence). "
        "Focus on defining traits that would help a bot interact with them.\n\n"
        f"Messages:\n{text_history}"
    )
    
    try:
        model = await _get_model()
        response = await model.generate_content_async(prompt)
        return response.text.strip()
    except Exception:
        # If it fails again, try to list models to help debugging
        try:
            available = [m.name for m in genai.list_models()]
            logger.error(f"Failed to generate content. Available models: {available}")
        except:
            pass
        logger.exception("Failed to summarize user history via Gemini")
        return None

async def generate_personalized_reply(profile: str, current_message: str) -> str | None:
    """Generate a response based on the user's profile and current message."""
    prompt = (
        f"You are a helpful and funny bot in a Telegram chat. "
        f"Here is what we know about the user: {profile}. "
        f"They just said: \"{current_message}\". "
        f"Generate a short, characteristic response to this message (max 2 sentences). "
        f"The response should feel like you've known them for a while."
    )
    
    try:
        model = await _get_model()
        response = await model.generate_content_async(prompt)
        return response.text.strip()
    except Exception:
        logger.exception("Failed to generate personalized reply via Gemini")
        return None
