from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests
import streamlit as st


API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
CHATS_DIR = Path("chats")
MEMORY_PATH = Path("memory.json")
SYSTEM_PROMPT = "You are a helpful assistant."


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def format_timestamp(timestamp: str) -> str:
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return parsed.strftime("%b %d, %Y %I:%M %p")
    except ValueError:
        return timestamp


def chat_file_path(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}.json"


def derive_title(messages: list[dict[str, str]]) -> str:
    for message in messages:
        if message.get("role") == "user" and message.get("content", "").strip():
            title = message["content"].strip().splitlines()[0]
            return title[:40] + ("..." if len(title) > 40 else "")
    return "New chat"


def new_chat() -> dict[str, Any]:
    timestamp = utc_now_iso()
    chat_id = uuid4().hex
    return {
        "chat_id": chat_id,
        "title": "New chat",
        "created_at": timestamp,
        "updated_at": timestamp,
        "messages": [],
    }


def save_chat(chat: dict[str, Any]) -> None:
    CHATS_DIR.mkdir(exist_ok=True)
    chat["title"] = derive_title(chat["messages"])
    chat["updated_at"] = utc_now_iso()
    chat_file_path(chat["chat_id"]).write_text(
        json.dumps(chat, indent=2),
        encoding="utf-8",
    )


def delete_chat_file(chat_id: str) -> None:
    path = chat_file_path(chat_id)
    if path.exists():
        path.unlink()


def load_chats() -> dict[str, dict[str, Any]]:
    CHATS_DIR.mkdir(exist_ok=True)
    chats: dict[str, dict[str, Any]] = {}
    for path in CHATS_DIR.glob("*.json"):
        try:
            chat = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        chat_id = chat.get("chat_id")
        if not chat_id:
            continue
        chat.setdefault("title", "New chat")
        chat.setdefault("created_at", utc_now_iso())
        chat.setdefault("updated_at", chat["created_at"])
        chat.setdefault("messages", [])
        chats[chat_id] = chat
    return dict(
        sorted(
            chats.items(),
            key=lambda item: item[1].get("updated_at", ""),
            reverse=True,
        )
    )


def load_memory() -> dict[str, Any]:
    if not MEMORY_PATH.exists():
        return {}
    try:
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_memory(memory: dict[str, Any]) -> None:
    MEMORY_PATH.write_text(json.dumps(memory, indent=2), encoding="utf-8")


def reset_memory() -> None:
    st.session_state.memory = {}
    save_memory(st.session_state.memory)


def normalize_memory_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        cleaned = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text and text not in cleaned:
                    cleaned.append(text)
        return cleaned
    return value


def merge_memory(existing: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in updates.items():
        normalized = normalize_memory_value(value)
        if normalized in ("", [], None):
            continue

        current = merged.get(key)
        if isinstance(current, list) and isinstance(normalized, list):
            merged[key] = list(dict.fromkeys([*current, *normalized]))
        elif isinstance(current, list) and isinstance(normalized, str):
            merged[key] = list(dict.fromkeys([*current, normalized]))
        elif isinstance(current, str) and isinstance(normalized, list):
            merged[key] = list(dict.fromkeys([current, *normalized]))
        else:
            merged[key] = normalized
    return merged


def format_memory_for_prompt(memory: dict[str, Any]) -> str:
    if not memory:
        return ""
    return json.dumps(memory, indent=2)


def init_state() -> None:
    if "chats" not in st.session_state:
        st.session_state.chats = load_chats()
    if "active_chat_id" not in st.session_state:
        chat_ids = list(st.session_state.chats.keys())
        st.session_state.active_chat_id = chat_ids[0] if chat_ids else None
    if "memory" not in st.session_state:
        st.session_state.memory = load_memory()


def create_chat(make_active: bool = True) -> str:
    chat = new_chat()
    st.session_state.chats[chat["chat_id"]] = chat
    save_chat(chat)
    reorder_chats()
    if make_active:
        st.session_state.active_chat_id = chat["chat_id"]
    return chat["chat_id"]


def reorder_chats() -> None:
    st.session_state.chats = dict(
        sorted(
            st.session_state.chats.items(),
            key=lambda item: item[1].get("updated_at", ""),
            reverse=True,
        )
    )


def set_active_chat(chat_id: str) -> None:
    st.session_state.active_chat_id = chat_id


def remove_chat(chat_id: str) -> None:
    st.session_state.chats.pop(chat_id, None)
    delete_chat_file(chat_id)
    if st.session_state.active_chat_id == chat_id:
        remaining_ids = list(st.session_state.chats.keys())
        st.session_state.active_chat_id = remaining_ids[0] if remaining_ids else None
    reorder_chats()


def get_hf_token() -> str | None:
    token = st.secrets.get("HF_TOKEN", "")
    return token.strip() or None


def build_api_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    memory_text = format_memory_for_prompt(st.session_state.memory)
    system_content = SYSTEM_PROMPT
    if memory_text:
        system_content += (
            "\n\nKnown user memory for personalization:\n"
            f"{memory_text}\n"
            "Use this memory only when it is relevant to the user's request."
        )
    return [{"role": "system", "content": system_content}, *messages]


def explain_http_error(exc: requests.HTTPError) -> str:
    response = exc.response
    if response is None:
        return "The request failed before the API returned a response."
    if response.status_code == 401:
        return "Your Hugging Face token was rejected. Check `HF_TOKEN` in Streamlit secrets."
    if response.status_code == 429:
        return "The API rate limit was reached. Please wait a moment and try again."
    if response.status_code >= 500:
        return "The Hugging Face service is having trouble right now. Please try again shortly."
    body = response.text.strip()
    return f"API error {response.status_code}: {body or 'No error details were returned.'}"


def request_json_completion(messages: list[dict[str, str]], hf_token: str) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 160,
    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"].strip()
    parsed = json.loads(content)
    return parsed if isinstance(parsed, dict) else {}


def extract_user_memory(user_text: str, hf_token: str) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "Extract stable user traits or preferences from the user's message. "
                "Return only a JSON object. "
                "Use short keys like name, interests, favorite_topics, preferred_language, "
                "communication_style, location, or goals when relevant. "
                "If nothing useful is present, return {}."
            ),
        },
        {
            "role": "user",
            "content": user_text,
        },
    ]
    return request_json_completion(messages, hf_token)


def extract_stream_text(event: dict[str, Any]) -> str:
    choices = event.get("choices") or []
    if not choices:
        return ""

    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        )
    return ""


def stream_chat_completion(messages: list[dict[str, str]], hf_token: str):
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": build_api_messages(messages),
        "max_tokens": 512,
        "stream": True,
    }
    with requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=60,
        stream=True,
    ) as response:
        response.raise_for_status()

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue

            data = raw_line.removeprefix("data:").strip()
            if data == "[DONE]":
                break

            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            chunk = extract_stream_text(event)
            if chunk:
                yield chunk
                time.sleep(0.02)


def render_sidebar() -> None:
    with st.sidebar:
        st.title("My AI Chat")
        if st.button("New Chat", use_container_width=True, type="primary"):
            create_chat(make_active=True)
            st.rerun()

        with st.expander("User Memory", expanded=True):
            if st.session_state.memory:
                st.json(st.session_state.memory)
            else:
                st.caption("No saved memory yet.")
            if st.button("Clear Memory", use_container_width=True):
                reset_memory()
                st.rerun()

        st.caption("Saved chats")

        for chat_id, chat in st.session_state.chats.items():
            is_active = chat_id == st.session_state.active_chat_id
            title = chat.get("title", "New chat")
            updated_at = format_timestamp(chat.get("updated_at", ""))
            row = st.columns([5, 1], gap="small")

            if row[0].button(
                f"{title}\n{updated_at}",
                key=f"open_{chat_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                set_active_chat(chat_id)
                st.rerun()

            if row[1].button("X", key=f"delete_{chat_id}", use_container_width=True):
                remove_chat(chat_id)
                st.rerun()


def render_empty_state(hf_token: str | None) -> None:
    st.info("Start a new conversation from the input bar below, or run a quick API test.")
    if st.button('Send test message: "Hello!"'):
        if st.session_state.active_chat_id is None:
            create_chat(make_active=True)
        with st.chat_message("user"):
            st.write("Hello!")
        send_message("Hello!", hf_token)
        st.rerun()


def render_chat(chat: dict[str, Any]) -> None:
    if not chat["messages"]:
        st.subheader("New conversation")
        st.caption("Messages will appear here and stay attached to this chat.")
    for message in chat["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def send_message(user_text: str, hf_token: str | None) -> None:
    active_chat_id = st.session_state.active_chat_id
    if active_chat_id is None:
        active_chat_id = create_chat(make_active=True)

    chat = st.session_state.chats[active_chat_id]
    chat["messages"].append({"role": "user", "content": user_text})
    save_chat(chat)
    reorder_chats()

    if not hf_token:
        st.error(
            "Missing Hugging Face token. Add `HF_TOKEN` to `.streamlit/secrets.toml` "
            "or Streamlit Community Cloud secrets."
        )
        return

    try:
        with st.chat_message("assistant"):
            assistant_text = st.write_stream(
                stream_chat_completion(chat["messages"], hf_token)
            )
    except requests.HTTPError as exc:
        st.error(explain_http_error(exc))
        return
    except requests.RequestException as exc:
        st.error(f"Network error: {exc}")
        return
    except (KeyError, IndexError, TypeError, ValueError):
        st.error("The API returned an unexpected response format.")
        return

    if not isinstance(assistant_text, str):
        assistant_text = "".join(assistant_text)

    if not assistant_text.strip():
        st.error("The API stream completed without returning any assistant text.")
        return

    chat["messages"].append({"role": "assistant", "content": assistant_text})
    save_chat(chat)
    reorder_chats()

    try:
        extracted_memory = extract_user_memory(user_text, hf_token)
    except requests.HTTPError:
        return
    except requests.RequestException:
        return
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return

    st.session_state.memory = merge_memory(st.session_state.memory, extracted_memory)
    save_memory(st.session_state.memory)


st.set_page_config(page_title="My AI Chat", layout="wide")
init_state()

hf_token = get_hf_token()

render_sidebar()

if st.session_state.active_chat_id is None and not st.session_state.chats:
    create_chat(make_active=True)

active_chat_id = st.session_state.active_chat_id
active_chat = st.session_state.chats.get(active_chat_id) if active_chat_id else None

st.title("My AI Chat")
st.caption(f"Model: {MODEL_NAME}")

if not hf_token:
    st.error(
        "Missing Hugging Face token. Add `HF_TOKEN` to `.streamlit/secrets.toml` "
        "or Streamlit Community Cloud secrets."
    )

if active_chat is None:
    st.info("No chats yet. Create one from the sidebar to get started.")
else:
    render_chat(active_chat)
    if not active_chat["messages"]:
        render_empty_state(hf_token)

if prompt := st.chat_input("Message the assistant"):
    with st.chat_message("user"):
        st.write(prompt)
    send_message(prompt, hf_token)
    st.rerun()
