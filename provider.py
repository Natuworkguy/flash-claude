"""Claude, as a model provider for Flash.

Flash runs this once per request, with JSON on stdin (see Flash's
docs/EXTENSIONS.md, Providers). Flash's models are named @claude/<model>,
such as @claude/opus-5-5, which this turns into the API's claude-opus-5-5.

Flash sends Ollama's shapes; this turns them into Anthropic's Messages
API and the answer back. Claude's replies are replayed to it exactly as
it wrote them, thinking blocks and tool ids included, from a cache kept
in this extension's folder: Flash keeps only the text and the tool calls
of a reply, and the API wants the rest back on the next request.

It talks to the API with the official SDK, installed into this
extension's own folder the first time it is needed, so nothing is added
to the Python Flash runs on.

The API key is this extension's own setting (API key, on Flash's
Extensions page, or /extension set claude api_key <key>), handed to it
as FLASH_SETTING_API_KEY. Left empty, the environment's is used.

Settings, from the environment (/set NAME VALUE in Flash):
  ANTHROPIC_API_KEY  the key, unless the setting has one, or `ant auth
                     login` signed you in
  CLAUDE_EFFORT      low, medium, high (the default), xhigh or max; a
                     reply Flash asks to think harder is always max
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import io
import json
import os
import re
import shutil
import struct
import subprocess  # nosec B404 -- installing the SDK, once
import sys
from pathlib import Path
from typing import Any

HERE = Path(os.environ.get("FLASH_EXTENSION_DIR") or Path(__file__).parent)
DEPS = HERE / ".deps"
PILLOW = HERE / ".pillow"
CACHE = HERE / ".replies"
KEEP_REPLIES = 2000

# The largest picture the API takes: no side over 8000 pixels, or over
# 2000 once a request has more than 20 pictures, and 5 MB of base64.
# A larger one, such as a full-page screenshot, is shrunk to fit.
MAX_SIDE = 8000
MAX_SIDE_MANY = 2000
MANY_IMAGES = 20
MAX_IMAGE_BYTES = 5 * 1024 * 1024

# Streamed, so a long answer never runs into an HTTP timeout.
MAX_TOKENS = 64000

EFFORTS = ("low", "medium", "high", "xhigh", "max")

# Shown when the API cannot be asked which models there are: the current
# ones, each with a 1M-token window.
KNOWN = ("opus-5-5", "fable-5-1", "sonnet-5-5", "haiku-5-5")
KNOWN_CONTEXT = 1_000_000

# What every current Claude model does, in Ollama's words.
CAPABILITIES = ["completion", "tools", "vision", "thinking"]

# Models that take the server-side refusal fallback: a declined request
# is run again on a model that can take it, inside the same call.
FALLBACK_MODELS = {
    "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5",
    "claude-sonnet-5-5",
}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

NO_KEY = (
    "Give it an API key: in its settings on Flash's Extensions page, with "
    "/extension set claude api_key sk-ant-..., or /set ANTHROPIC_API_KEY "
    "sk-ant-.... Or sign in with `ant auth login`."
)

# A reply Flash relays as a note to the model rather than its own turn.
NOTE = "[A note from Flash]"


def api_key() -> str | None:
    """The key from this extension's settings, which goes ahead of
    ANTHROPIC_API_KEY; None to leave the SDK to find one."""

    return os.environ.get("FLASH_SETTING_API_KEY", "").strip() or None


def client_of(anthropic, **options):
    key = api_key()
    return anthropic.Anthropic(**({"api_key": key} if key else {}),
                               **options)


def say(answer: dict) -> None:
    print(json.dumps(answer), flush=True)


# Names -----------------------------------------------------------------------


def api_name(model: str) -> str:
    """Flash's opus-5-5 is the API's claude-opus-5-5, and so is opus-5.5:
    a version may be written with a dot, as the models' names are."""

    model = re.sub(r"(?<=\d)\.(?=\d)", "-", model.strip())
    return model if model.startswith("claude-") else f"claude-{model}"


def flash_name(model: str) -> str:
    return model[len("claude-"):] if model.startswith("claude-") else model


def _version(model: str) -> tuple[int, int]:
    """claude-opus-4-6 is (4, 6); a dated or older id, (0, 0)."""

    found = re.match(r"claude-[a-z]+-(\d+)(?:-(\d{1,2}))?$", model)
    if not found:
        return (0, 0)
    return (int(found.group(1)), int(found.group(2) or 0))


def adaptive(model: str) -> bool:
    """Whether MODEL takes adaptive thinking and effort: 4.6 and later.
    Older ones are sent neither, and run as the API sets them."""

    return _version(model) >= (4, 6)


# The SDK, in this extension's folder -----------------------------------------


def _installed(module: str, package: str, folder: Path):
    """MODULE, with PACKAGE installed into FOLDER beside this file on
    first use."""

    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    try:
        return importlib.import_module(module)
    except ImportError:
        pass

    incoming = folder.with_name(folder.name + "-incoming")
    shutil.rmtree(incoming, ignore_errors=True)
    tries = [
        [sys.executable, "-m", "pip", "install", "--quiet",
         "--disable-pip-version-check", "--target", str(incoming),
         package],
        # A uv environment has no pip of its own.
        ["uv", "pip", "install", "--quiet", "--target", str(incoming),
         "--python", sys.executable, package],
    ]
    why = ""
    for argv in tries:
        try:
            done = subprocess.run(  # nosec B603
                argv, capture_output=True, text=True, timeout=240,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            why = str(exc)
            continue
        if done.returncode == 0:
            shutil.rmtree(folder, ignore_errors=True)
            incoming.rename(folder)
            # Python looked for the folder before it was there, and
            # remembers that it was not.
            sys.path_importer_cache.pop(str(folder), None)
            importlib.invalidate_caches()
            return importlib.import_module(module)
        why = (done.stderr or done.stdout).strip().splitlines()[-1:] or [""]
        why = why[0]
    raise RuntimeError(
        f"could not install the {package} package for the Claude provider"
        + (f" ({why})" if why else "")
        + f". Install it into the Python Flash runs on: {sys.executable} "
        f"-m pip install {package}"
    )


def sdk():
    """The anthropic package, installed beside this file on first use."""

    return _installed("anthropic", "anthropic", DEPS)


def pillow():
    """Pillow's Image, installed beside this file the first time a
    picture is too large to send as it is."""

    image = _installed("PIL.Image", "pillow", PILLOW)
    # A full-page screenshot can be taller than Pillow trusts by default;
    # it is the user's own picture, not a decompression bomb.
    image.MAX_IMAGE_PIXELS = None
    return image


# Replies kept for replay -----------------------------------------------------


def _key(content: str, calls: list) -> str:
    """A reply, known by the text and tool calls Flash keeps of it."""

    shape = {
        "content": (content or "").strip(),
        "calls": [
            [c.get("name", ""), c.get("arguments") or {}] for c in calls
        ],
    }
    return hashlib.sha256(
        json.dumps(shape, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _calls_of(message: dict) -> list:
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {}
        calls.append({"name": function.get("name", ""),
                      "arguments": arguments})
    return calls


def keep(content: str, calls: list, blocks: list) -> None:
    """Keep BLOCKS, the reply as Claude wrote it, for its replay."""

    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / f"{_key(content, calls)}.json").write_text(
            json.dumps(blocks), encoding="utf-8",
        )
        kept = sorted(CACHE.glob("*.json"), key=lambda p: p.stat().st_mtime)
        for old in kept[:-KEEP_REPLIES]:
            old.unlink(missing_ok=True)
    except OSError:
        pass


def kept(content: str, calls: list) -> list | None:
    try:
        found = json.loads(
            (CACHE / f"{_key(content, calls)}.json").read_text("utf-8")
        )
    except (OSError, ValueError):
        return None
    return found if isinstance(found, list) else None


# Flash's messages as Anthropic's ---------------------------------------------


def _media_type(data: str) -> str:
    head = base64.b64decode(data[:32] + "=" * (-len(data[:32]) % 4))
    if head.startswith(b"\x89PNG"):
        return "image/png"
    if head.startswith(b"GIF8"):
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _size(raw: bytes) -> tuple[int, int] | None:
    """A picture's width and height, read from its header; None for a
    kind this does not know."""

    if raw.startswith(b"\x89PNG") and len(raw) >= 24:
        return struct.unpack(">II", raw[16:24])
    if raw.startswith(b"GIF8") and len(raw) >= 10:
        return struct.unpack("<HH", raw[6:10])
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP" and len(raw) >= 30:
        chunk = raw[12:16]
        if chunk == b"VP8X":
            return (1 + int.from_bytes(raw[24:27], "little"),
                    1 + int.from_bytes(raw[27:30], "little"))
        if chunk == b"VP8L":
            bits = int.from_bytes(raw[21:25], "little")
            return (1 + (bits & 0x3FFF), 1 + ((bits >> 14) & 0x3FFF))
        if chunk == b"VP8 ":
            width, height = struct.unpack("<HH", raw[26:30])
            return (width & 0x3FFF, height & 0x3FFF)
        return None
    if raw.startswith(b"\xff\xd8"):
        # A JPEG's size is in its frame header, after any other segments.
        at = 2
        while at + 9 <= len(raw) and raw[at] == 0xFF:
            marker = raw[at + 1]
            if marker == 0xFF or marker == 0x01 or 0xD0 <= marker <= 0xD9:
                at += 1 if marker == 0xFF else 2
                continue
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                height, width = struct.unpack(">HH", raw[at + 5:at + 9])
                return (width, height)
            at += 2 + int.from_bytes(raw[at + 2:at + 4], "big")
        return None
    return None


def fit(data: str, side: int) -> str:
    """DATA, a picture in base64, as the API takes it: no side longer
    than SIDE, and no more than MAX_IMAGE_BYTES. One that fits already
    is left as it is. ValueError when it cannot be made to fit."""

    try:
        raw = base64.b64decode(data)
    except ValueError:
        return data  # the API says what is wrong with it
    size = _size(raw)
    if (size is None or max(size) <= side) and len(data) <= MAX_IMAGE_BYTES:
        return data

    try:
        image = pillow()
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    try:
        with image.open(io.BytesIO(raw)) as opened:
            kind = opened.format
            picture = opened.convert("RGBA") if opened.mode not in (
                "RGB", "RGBA", "L", "LA",
            ) else opened.copy()
    except Exception as exc:  # noqa: BLE001 -- Pillow could not read it
        raise ValueError(f"it could not be read ({exc})") from exc

    # A screenshot stays a lossless PNG unless that is too many bytes.
    form = kind if kind in ("PNG", "JPEG", "WEBP") else "PNG"
    picture.thumbnail((side, side), image.LANCZOS)
    while True:
        out = io.BytesIO()
        if form == "JPEG":
            picture.convert("RGB").save(out, "JPEG", quality=85)
        else:
            picture.save(out, form)
        shrunk = base64.b64encode(out.getvalue()).decode("ascii")
        if len(shrunk) <= MAX_IMAGE_BYTES:
            return shrunk
        if form != "JPEG":
            form = "JPEG"
            continue
        longest = max(picture.size)
        if longest <= 64:
            raise ValueError("it is too many bytes even when small")
        picture.thumbnail((longest * 3 // 4,) * 2, image.LANCZOS)


def _images(message: dict, side: int = MAX_SIDE) -> list:
    blocks = []
    for data in message.get("images") or []:
        try:
            data = fit(data, side)
        except ValueError as exc:
            blocks.append({"type": "text", "text": (
                "(A picture was left out: it is larger than Claude takes, "
                f"and could not be shrunk to fit: {exc})"
            )})
            continue
        blocks.append({"type": "image", "source": {
            "type": "base64", "media_type": _media_type(data), "data": data,
        }})
    return blocks


def tools_of(tools: list | None) -> list:
    """Ollama's tool schemas as Anthropic's."""

    found = []
    for tool in tools or []:
        function = tool.get("function") or tool
        if not function.get("name"):
            continue
        found.append({
            "name": function["name"],
            "description": function.get("description") or "",
            "input_schema": function.get("parameters")
            or {"type": "object", "properties": {}},
        })
    return found


def convert(messages: list, flat: bool = False) -> tuple[str, list]:
    """MESSAGES, in Ollama's shape, as Anthropic's system and messages.

    FLAT turns every tool call and result into plain text: for a history
    whose replies Claude did not write here, when the API will not take
    them as tool turns without the thinking that came with them."""

    system: list[str] = []
    out: list[dict] = []
    pictures = sum(
        len(m.get("images") or []) for m in messages
        if m.get("role") not in ("system", "tool", "assistant")
    )
    side = MAX_SIDE_MANY if pictures > MANY_IMAGES else MAX_SIDE
    waiting: list[tuple[str, str]] = []  # tool_use ids not yet answered

    def user(blocks: list) -> None:
        if not blocks:
            return
        if out and out[-1]["role"] == "user":
            out[-1]["content"] += blocks
        else:
            out.append({"role": "user", "content": blocks})

    def settle() -> None:
        """Answer any tool call left unanswered: the API wants every
        tool_use followed by its result."""

        if waiting:
            user([{"type": "tool_result", "tool_use_id": tid,
                   "content": "The tool was not run.", "is_error": True}
                  for tid, _ in waiting])
            waiting.clear()

    for index, message in enumerate(messages):
        role = message.get("role")
        text = message.get("content") or ""
        if not isinstance(text, str):
            text = json.dumps(text)

        if role == "system":
            if not out:
                system.append(text)
            else:
                settle()
                user([{"type": "text", "text": f"{NOTE}\n{text}"}])
            continue

        if role == "tool":
            name = message.get("tool_name") or "the tool"
            if flat or not waiting:
                settle()
                user([{"type": "text",
                       "text": f"[{name} returned]\n{text}"}])
                continue
            at = next((i for i, (_, n) in enumerate(waiting) if n == name), 0)
            tid, _ = waiting.pop(at)
            user([{"type": "tool_result", "tool_use_id": tid,
                   "content": text or "(no output)"}])
            continue

        if role == "assistant":
            settle()
            calls = _calls_of(message)
            blocks = None if flat else kept(text, calls)
            if blocks is None:
                blocks = [{"type": "text", "text": text}] if text.strip() \
                    else []
                for j, call in enumerate(calls):
                    if flat:
                        blocks.append({"type": "text", "text": (
                            f"[I called {call['name']} with "
                            f"{json.dumps(call['arguments'])}]"
                        )})
                    else:
                        blocks.append({
                            "type": "tool_use",
                            "id": f"toolu_flash_{index}_{j}",
                            "name": call["name"],
                            "input": call["arguments"],
                        })
            if not blocks:
                continue
            if not out:
                out.append({"role": "user", "content": [
                    {"type": "text", "text": "(The conversation so far.)"},
                ]})
            out.append({"role": "assistant", "content": blocks})
            waiting.extend(
                (b["id"], b["name"]) for b in blocks
                if b.get("type") == "tool_use"
            )
            continue

        # A user's message, with any pictures before its words.
        settle()
        blocks = _images(message, side)
        if text.strip():
            blocks.append({"type": "text", "text": text})
        user(blocks)

    settle()
    return "\n\n".join(s for s in system if s.strip()), out


# Anthropic's answer as Flash's -----------------------------------------------


def _blocks(message) -> list[dict]:
    """The reply's blocks as plain JSON, those worth replaying."""

    kept_types = ("text", "thinking", "redacted_thinking", "tool_use")
    found = []
    for block in message.content:
        if getattr(block, "type", "") in kept_types:
            found.append(block.model_dump(mode="json", exclude_none=True))
    return found


def answer_of(message) -> dict:
    """What Flash is told: the words, the summarized thinking, the tool
    calls, and what it cost. The reply as Claude wrote it is kept."""

    blocks = _blocks(message)
    text = "".join(b.get("text", "") for b in blocks if b["type"] == "text")
    thinking = "".join(
        b.get("thinking", "") for b in blocks if b["type"] == "thinking"
    )
    calls = [
        {"name": b["name"], "arguments": b.get("input") or {}}
        for b in blocks if b["type"] == "tool_use"
    ]
    stop = getattr(message, "stop_reason", "")

    if stop == "max_tokens" and calls:
        # A tool call cut off mid-input is never run.
        calls = []
        blocks = [b for b in blocks if b["type"] != "tool_use"]
        text += "\n\n(The reply ran out of room before its tool call was " \
            "complete.)"
    if stop == "refusal":
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        text = (text + "\n\n" if text else "") + (
            "Claude declined to continue this request"
            + (f" ({category})." if category else ".")
        )

    keep(text, calls, blocks)

    usage = getattr(message, "usage", None)
    read = sum(
        getattr(usage, name, 0) or 0 for name in (
            "input_tokens", "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
    ) if usage else 0
    reply: dict[str, Any] = {"role": "assistant", "content": text}
    if thinking:
        reply["thinking"] = thinking
    if calls:
        reply["tool_calls"] = [{"function": call} for call in calls]
    return {
        "message": reply,
        "prompt_eval_count": read,
        "eval_count": getattr(usage, "output_tokens", 0) if usage else 0,
        "done": True,
        "done_reason": stop or "stop",
    }


# The two requests ------------------------------------------------------------


def models() -> None:
    """The models there are: asked of the API when the SDK is here and
    the key works, else the current ones."""

    listed = []
    if (DEPS / "anthropic").is_dir():
        try:
            client = client_of(sdk(), max_retries=0, timeout=10.0)
            for model in client.models.list():
                if not model.id.startswith("claude-"):
                    continue
                context = getattr(model, "max_input_tokens", None)
                listed.append({
                    "name": flash_name(model.id),
                    "capabilities": CAPABILITIES,
                    **({"context": context} if context else {}),
                })
        except Exception:  # noqa: BLE001 -- offline, or no key yet
            listed = []
    if not listed:
        listed = [
            {"name": name, "capabilities": CAPABILITIES,
             "context": KNOWN_CONTEXT}
            for name in KNOWN
        ]
    say({"models": listed})


def request_of(ask: dict, flat: bool = False) -> dict:
    model = api_name(ask["model"])
    system, messages = convert(ask.get("messages") or [], flat)
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "messages": messages,
        # Caches the conversation so far: an agent's next step resends it.
        "cache_control": {"type": "ephemeral"},
    }
    if system:
        params["system"] = system
    tools = tools_of(ask.get("tools"))
    if tools:
        params["tools"] = tools
    if adaptive(model):
        # Its reasoning summarized, for Flash to show as it does an Ollama
        # model's thinking.
        params["thinking"] = {"type": "adaptive", "display": "summarized"}
        effort = os.environ.get("CLAUDE_EFFORT", "").strip().lower()
        effort = effort if effort in EFFORTS else "high"
        if ask.get("think"):
            # Flash asked it to think harder (Think harder, or Flash
            # deciding a message needs it): as hard as it goes.
            effort = "max"
        params["output_config"] = {"effort": effort}
    if model in FALLBACK_MODELS:
        params["betas"] = [FALLBACK_BETA]
        params["fallbacks"] = "default"
    return params


def _error(anthropic, exc) -> dict:
    """What went wrong, said so the user knows what to do about it."""

    if isinstance(exc, anthropic.AuthenticationError):
        return {"error": "Claude did not accept the API key. " + NO_KEY,
                "status": 401}
    if isinstance(exc, anthropic.PermissionDeniedError):
        return {"error": f"Not allowed: {exc.message}", "status": 403}
    if isinstance(exc, anthropic.NotFoundError):
        return {"error": f"No such model, or no access to it: {exc.message}",
                "status": 404}
    if isinstance(exc, anthropic.RateLimitError):
        wait = exc.response.headers.get("retry-after", "")
        return {"error": "Rate limited"
                + (f": try again in {wait}s." if wait else "."),
                "status": 429}
    if isinstance(exc, anthropic.APIStatusError):
        return {"error": exc.message, "status": exc.status_code}
    if isinstance(exc, anthropic.APIConnectionError):
        return {"error": "Could not reach the Anthropic API. Check the "
                         "connection.", "status": 503}
    return {"error": str(exc)}


def chat(ask: dict) -> None:
    try:
        anthropic = sdk()
    except RuntimeError as exc:
        say({"error": str(exc)})
        return

    client = client_of(anthropic)
    stream_it = bool(ask.get("stream"))
    streamed: list[str] = []

    def run(flat: bool):
        params = request_of(ask, flat)
        start = client.beta.messages.stream if "betas" in params \
            else client.messages.stream
        with start(**params) as stream:
            if stream_it:
                for event in stream:
                    if event.type != "content_block_delta":
                        continue
                    if event.delta.type == "text_delta":
                        streamed.append(event.delta.text)
                        say({"message": {"content": event.delta.text},
                             "done": False})
                    elif event.delta.type == "thinking_delta":
                        say({"message": {"thinking": event.delta.thinking},
                             "done": False})
            return stream.get_final_message()

    try:
        try:
            final = run(flat=False)
        except anthropic.BadRequestError as exc:
            # A history Claude did not write here, whose tool turns the
            # API will not take without their thinking: once more, as
            # plain text.
            if "thinking" not in str(exc.message).lower():
                raise
            final = run(flat=True)
    except anthropic.APIError as exc:
        say(_error(anthropic, exc))
        return
    except TypeError as exc:
        # No key and no `ant auth login`: the SDK says so before asking.
        if "authentication" not in str(exc).lower():
            raise
        say({"error": NO_KEY, "status": 401})
        return

    whole = answer_of(final)
    if stream_it:
        # The words and thinking went already: only what was added to
        # them since, a note on a cut-off or a declined reply, is left.
        sent = "".join(streamed)
        text = whole["message"]["content"]
        whole["message"] = {
            key: value for key, value in whole["message"].items()
            if key not in ("content", "thinking")
        }
        whole["message"]["content"] = (
            text[len(sent):] if text.startswith(sent) else ""
        )
    say(whole)


def main() -> None:
    ask = json.load(sys.stdin)
    if ask.get("action") == "models":
        models()
    elif ask.get("action") == "chat":
        chat(ask)
    else:
        say({"error": f"unknown action {ask.get('action')!r}"})


if __name__ == "__main__":
    main()
