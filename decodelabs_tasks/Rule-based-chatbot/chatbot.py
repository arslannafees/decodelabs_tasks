from typing import Tuple
BOT_NAME = "Nova"
INTENT_KEYWORDS = {
    "greeting": {"hello": True, "hi": True, "hey": True, "greetings": True},
    "farewell": {"bye": True, "quit": True, "exit": True, "goodbye": True},
    "help": {"help": True, "commands": True, "menu": True},
    "thanks": {"thanks": True, "thank": True, "thankyou": True, "appreciate": True},
    "name": {"name": True, "whoareyou": True, "who": True, "bot": True},
    "capability": {"capability": True, "abilities": True, "can": True, "do": True},
}
INTENT_RESPONSES = {
    "greeting": f"Hello! I am {BOT_NAME}. How can I help you today?",
    "farewell": f"Goodbye from {BOT_NAME}! Stay safe and keep learning.",
    "help": (
        "I can respond to greetings, help requests, thanks, my name, and exit commands. "
        "Try: hello, help, thanks, who are you, or bye."
    ),
    "thanks": "You are welcome. I am here to stay deterministic and traceable.",
    "name": f"My name is {BOT_NAME}, a rule-based white-box chatbot.",
    "capability": "I only use explicit rules, dictionary lookup, and a fallback response.",
}
FALLBACK_RESPONSE = "I do not understand. Please try a supported keyword like hello or help."
def build_help_response(clean_input: str) -> str:
    """Nested conditions: refine help answers using a second decision layer."""
    if "project" in clean_input or "report" in clean_input:
        return (
            "Help path: you can ask about a project, a report, or request a summary. "
            "Example: 'help projects' or 'help reports'."
        )
    if "feature" in clean_input or "capability" in clean_input:
        return (
            "Help path: I support greeting, farewell, help, thanks, name, and capability queries."
        )
    return (
        "Help path: try greetings, exit commands, thanks, or ask who I am. "
        "Example: 'hello', 'bye', 'thanks', 'who are you'."
    )
def resolve_intent(clean_input: str) -> str | None:
    """Process phase: resolve the first matching intent with dictionary lookup."""
    if not clean_input:
        return None
    exact_intent = {
        "hello": "greeting",
        "hi": "greeting",
        "hey": "greeting",
        "greetings": "greeting",
        "bye": "farewell",
        "quit": "farewell",
        "exit": "farewell",
        "goodbye": "farewell",
        "help": "help",
        "commands": "help",
        "menu": "help",
        "thanks": "thanks",
        "thank you": "thanks",
        "thankyou": "thanks",
        "who are you": "name",
        "what is your name": "name",
        "name": "name",
        "can you do": "capability",
        "what can you do": "capability",
    }
    exact_match = exact_intent.get(clean_input)
    if exact_match:
        return exact_match
    words = clean_input.split()
    for word in words:
        for intent_name, keyword_set in INTENT_KEYWORDS.items():
            if keyword_set.get(word):
                return intent_name
    return None
def sanitize_input(raw: str) -> str:
    """Sanitization phase: normalize user input for deterministic matching."""
    return raw.lower().strip()
def process_input(clean_input: str) -> Tuple[str, bool]:
    """Logic Engine (Process): resolve intents with explicit dictionary rules.
    Returns a tuple of (response, should_exit).
    """
    intent = resolve_intent(clean_input)
    if intent == "help":
        return (build_help_response(clean_input), False)
    if intent == "farewell":
        return (INTENT_RESPONSES[intent], True)
    if intent is not None:
        return (INTENT_RESPONSES[intent], False)
    # Fallback: explicit, deterministic default response
    return (FALLBACK_RESPONSE, False)
def main() -> None:
    print(f"Deterministic Rule-Based Chatbot ({BOT_NAME}) — type 'exit' to quit")
    try:
        while True:
            raw = input("You: ")
            clean = sanitize_input(raw)
            response, should_exit = process_input(clean)
            print("Bot:", response)
            if should_exit:
                break
    except KeyboardInterrupt:
        print("\nexitting thankyou for using Nova AI")
if __name__ == "__main__":
    main()