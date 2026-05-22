# Rule based Chatbot

A small, deterministic rule-based chatbot implemented in pure Python. The bot uses explicit rules and a simple intent lookup to produce predictable, traceable responses — no machine learning models, no network calls.
Features
- Deterministic rule engine (white-box logic)
- Console interface plus an optional GUI wrapper
- Easy to read and extend — ideal for learning or prototypes

Requirements
- Python 3.8 or newer

Quick start
1. Open a terminal in the project folder `project1_rule_based_chatbot`.
2. Run the console bot:

```bash
python chatbot.py
```

Example conversation

```
You: Hi
Bot: Hello! How can I assist you today?

You: What's the weather?
Bot: I'm sorry, I don't understand. I operate only on explicit rules.

You: bye
Bot: Goodbye! Have a great day.
```

Extending the bot
- Add new intents and responses in `chatbot.py` using the existing lookup and decision structure.

License & Contributing
- Feel free to open issues or submit a PR with improvements.

Questions?
- Tell me if you want a README expanded with examples, tests, or packaging instructions.
