# Claude for Flash

A [Flash](https://github.com/Natuworkguy/Flash) extension that adds
Anthropic's Claude models as a model provider. Once it is installed,
Claude is picked like any other model:

```text
/model @claude/opus-5-5
```

It also shows in the web UI's model menu, can be a spark's model, and
runs sub-agents. Everything Flash does with a model works: tools,
images, thinking, and streaming in the web UI.

## Install

```text
/extension install github@Natuworkguy/flash-claude
```

The repository is private, so that works where git can already sign in
to GitHub as someone with access. Anywhere else, clone it and install
from the folder:

```bash
git clone https://github.com/Natuworkguy/flash-claude
flash --extension-install path@flash-claude
```

Then give it an API key from the Claude Console:

```text
/set ANTHROPIC_API_KEY sk-ant-...
```

If you signed in with `ant auth login`, there is nothing to set.

The first request installs Anthropic's Python SDK into the extension's
own folder, which takes a few seconds once. Nothing is added to the
Python Flash runs on.

## Models

`/model` lists what the API offers your key. Before the first request,
or offline, it lists the current models:

| Flash name | Model |
| --- | --- |
| `@claude/opus-5-5` | Claude Opus 5.5, the default choice |
| `@claude/fable-5-1` | Claude Fable 5.1, the most capable |
| `@claude/sonnet-5-5` | Claude Sonnet 5.5, quicker and cheaper |
| `@claude/haiku-5-5` | Claude Haiku 5.5, the fastest |

Every name is the API's model id without `claude-`, so any model your
key can use works, such as `@claude/opus-4-8`.

## Settings

Set with `/set NAME VALUE`, saved in `~/.flash.env`.

| Setting | Meaning |
| --- | --- |
| `ANTHROPIC_API_KEY` | Your key. Not needed after `ant auth login`. |
| `CLAUDE_EFFORT` | How hard Claude thinks: `low`, `medium`, `high` (the default), `xhigh` or `max`. Higher is slower and costs more; `xhigh` suits long coding work. |

## How it talks to Claude

- **Thinking** is adaptive, and its summary shows where Flash shows a
  model's thinking. Models older than 4.6 run without it.
- **Replies are replayed exactly.** Flash keeps only a reply's text and
  tool calls; the extension keeps the rest (thinking and tool ids) in
  `.replies/` in its folder, the newest 2000, and sends each reply back
  as Claude wrote it.
- **Prompt caching** is on, so each step of a long task re-reads the
  conversation so far at the cached price.
- **Refusal fallback** is on for the models that support it: if a
  safety check declines a request, the API runs it again on a model
  that can take it, in the same call.
- **Large pictures are shrunk to fit.** The API takes no side over
  8000 pixels (2000 once a request has more than 20 pictures) and no
  more than 5 MB, so a bigger one, such as a full-page screenshot, is
  scaled down first. That installs Pillow into the extension's folder
  the first time it is needed; if it cannot, the picture is left out
  with a note instead of failing the request.
- **A cut-off tool call is never run**: if a reply runs out of room in
  the middle of a tool call, Flash is told so instead.

Your conversations go to Anthropic's API, so `/stats` never counts these
models as running on your own machine.

## Development

```bash
python -m pytest
```

The tests use a stand-in for the SDK and never call the API. With Flash
installed in the same environment, one more test installs the extension
into a temporary Flash and lists its models.

MIT licensed. See [LICENSE](LICENSE).
