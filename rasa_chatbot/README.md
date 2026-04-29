# RASA Chatbot

Place your RASA project files here:

```
rasa_chatbot/
├── config.yml          # NLU pipeline and policy configuration
├── domain.yml          # Intents, entities, slots, responses, actions
├── data/
│   ├── nlu.yml         # Training examples for each intent
│   ├── stories.yml     # Conversation flow stories
│   └── rules.yml       # Rule-based dialogue rules
├── actions/
│   └── actions.py      # Custom action server (library lookup, mailing, etc.)
└── endpoints.yml       # Action server endpoint config
```

## Starting the RASA Server

```bash
# Train the model
rasa train

# Run the server with API enabled
rasa run --enable-api

# (Optional) Run the action server in a separate terminal
rasa run actions
```

The main application connects to `http://localhost:5005/webhooks/rest/webhook`.

## Supported Intents

- `greet`, `goodbye`, `affirm`, `deny`
- `ask_admission`, `ask_courses`, `ask_fees`
- `ask_staff`, `ask_hod`
- `ask_library` → triggers library book lookup
- `ask_navigate` → triggers campus navigation map
- `ask_placement`, `ask_exams`
- `ask_clubs`, `ask_societies`
- `book_appointment` → triggers mailing flow
- `ask_rules`, `ask_parking`, `ask_store`
