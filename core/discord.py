"""
Alfred AI - Discord Bot Integration
Connects Alfred agents to Discord channels.

Each channel maps to one agent. Each agent instance is isolated per channel
for thread safety (Agent has mutable conversation history).

Usage:
    from core.discord import DiscordBot
    bot = DiscordBot()
    bot.run()          # Foreground — Ctrl+C to stop
    bot.run_daemon()   # Background — writes PID file, logs to file
"""

import asyncio
import logging
import os
import signal
import sys
from typing import Optional
from pathlib import Path

import discord
from discord import Message, Thread

from .config import _load_config, config
from .agent import Agent, AgentConfig

logger = logging.getLogger("alfred.discord")

# PID and log file locations
PID_FILE = config.PROJECT_ROOT / "data" / "discord.pid"
LOG_FILE = config.PROJECT_ROOT / "data" / "discord.log"

MAX_MESSAGE_LENGTH = 2000  # Discord's hard limit


def _quiet_noisy_loggers():
    """Suppress noisy HTTP loggers from HuggingFace, httpx, etc."""
    for name in (
        "httpx",
        "httpcore",
        "urllib3",
        "sentence_transformers",
        "huggingface_hub",
        "transformers",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def is_bot_running() -> Optional[int]:
    """Check if the Discord bot is running. Returns PID if running, None otherwise."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        # Check if process is still alive
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        # Stale PID file — clean up
        PID_FILE.unlink(missing_ok=True)
        return None


def stop_bot() -> bool:
    """Stop the running Discord bot. Returns True if stopped, False if not running."""
    pid = is_bot_running()
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait briefly for clean shutdown
        import time
        for _ in range(20):  # 2 seconds max
            time.sleep(0.1)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
        PID_FILE.unlink(missing_ok=True)
        return True
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        return True
    except PermissionError:
        return False


class DiscordBot:
    """
    Discord bot that routes channel messages to Alfred agents.

    One Agent instance per channel for thread safety.
    Thread messages inherit their parent channel's agent.
    """

    def __init__(self):
        cfg = _load_config()
        self._discord_cfg = cfg.get("discord", {})

        if not self._discord_cfg.get("bot_token"):
            raise ValueError(
                "Discord bot token not configured. Run: alfred discord setup"
            )

        # discord.py client setup
        intents = discord.Intents.default()
        intents.message_content = True  # Required for reading message text
        intents.guilds = True
        intents.members = False

        self._client = discord.Client(intents=intents)
        self._guild_id = int(self._discord_cfg["guild_id"])

        # Channel ID -> config from alfred.json
        self._channel_configs: dict[int, dict] = {}
        for ch_id_str, ch_cfg in self._discord_cfg.get("channels", {}).items():
            self._channel_configs[int(ch_id_str)] = ch_cfg

        # Channel ID -> Agent instance (lazy-loaded, one per channel)
        self._agents: dict[int, Agent] = {}

        # Per-channel locks to prevent interleaved history corruption
        self._channel_locks: dict[int, asyncio.Lock] = {}

        # Register event handlers
        self._register_events()

    def _register_events(self):
        """Register discord.py event handlers."""

        @self._client.event
        async def on_ready():
            logger.info(
                f"Connected as {self._client.user} (ID: {self._client.user.id})"
            )
            guild = self._client.get_guild(self._guild_id)
            if guild:
                logger.info(f"Guild: {guild.name}")
                # Log channel mappings
                for ch_id, ch_cfg in self._channel_configs.items():
                    name = ch_cfg.get("name", ch_id)
                    agent = ch_cfg.get("agent", "?")
                    mention = "mention-only" if ch_cfg.get("require_mention", True) else "all messages"
                    logger.info(f"  #{name} -> {agent} ({mention})")
            else:
                logger.warning(f"Guild {self._guild_id} not found!")

            # Set presence
            activity = discord.Activity(
                type=discord.ActivityType.listening,
                name=f"{len(self._channel_configs)} channels",
            )
            await self._client.change_presence(activity=activity)

        @self._client.event
        async def on_message(message: Message):
            await self._handle_message(message)

    async def _handle_message(self, message: Message):
        """Process an incoming Discord message."""
        # 1. Ignore bot's own messages and other bots
        if message.author == self._client.user:
            return
        if message.author.bot:
            return

        # 2. Resolve effective channel ID (threads -> parent channel)
        effective_channel_id = self._resolve_channel_id(message)

        # 3. Check if this channel is configured
        if effective_channel_id not in self._channel_configs:
            return

        ch_cfg = self._channel_configs[effective_channel_id]

        # 4. Check require_mention
        if ch_cfg.get("require_mention", True):
            if not self._is_mentioned(message):
                return

        # 5. Extract clean message text (strip @mention)
        text = self._clean_message_text(message)
        if not text.strip():
            return

        # 6. Get per-channel lock
        if effective_channel_id not in self._channel_locks:
            self._channel_locks[effective_channel_id] = asyncio.Lock()

        async with self._channel_locks[effective_channel_id]:
            # 7. Get or create agent for this channel
            agent = self._get_agent(effective_channel_id)
            if agent is None:
                logger.error(f"No agent for channel {effective_channel_id}")
                return

            # 8. Process with typing indicator
            try:
                async with message.channel.typing():
                    # Run agent.run() in thread pool (it's synchronous)
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None, agent.run, text
                    )

                # 9. Send response (chunked if needed)
                await self._send_response(message.channel, response, reply_to=message)

            except Exception as e:
                logger.exception(
                    f"Error processing message in #{ch_cfg.get('name', effective_channel_id)}"
                )
                try:
                    await message.channel.send(
                        f"Sorry, I hit an error: {str(e)[:200]}"
                    )
                except discord.HTTPException:
                    pass

    def _resolve_channel_id(self, message: Message) -> int:
        """
        Resolve effective channel ID for agent routing.
        Thread messages inherit their parent channel's agent.
        """
        channel = message.channel
        if isinstance(channel, Thread) and channel.parent_id:
            return channel.parent_id
        return channel.id

    def _is_mentioned(self, message: Message) -> bool:
        """Check if the bot is @mentioned in this message."""
        if self._client.user in message.mentions:
            return True
        return False

    def _clean_message_text(self, message: Message) -> str:
        """
        Remove the bot's @mention from the message text.
        Discord formats mentions as <@BOT_ID> or <@!BOT_ID>.
        """
        text = message.content
        bot_id = str(self._client.user.id)
        text = text.replace(f"<@{bot_id}>", "").replace(f"<@!{bot_id}>", "")
        return text.strip()

    def _get_agent(self, channel_id: int) -> Optional[Agent]:
        """
        Get or create an Agent instance for a channel.
        Each channel gets its own instance for session isolation.
        """
        if channel_id not in self._agents:
            ch_cfg = self._channel_configs.get(channel_id)
            if not ch_cfg:
                return None

            agent_name = ch_cfg.get("agent")
            if not agent_name:
                return None

            try:
                cfg = _load_config()
                agent_data = cfg.get("agents", {}).get(agent_name)
                if not agent_data:
                    logger.error(f"Agent '{agent_name}' not found in alfred.json")
                    return None

                agent_data = dict(agent_data)  # Don't mutate the original
                agent_data["name"] = agent_name

                # Resolve workspace path
                workspace = Path(
                    agent_data.get("workspace", f"workspaces/{agent_name}")
                )
                if not workspace.is_absolute():
                    workspace = config.PROJECT_ROOT / workspace
                agent_data["workspace"] = str(workspace)

                agent_config = AgentConfig.from_dict(agent_data)

                # Ensure workspace dirs exist
                workspace.mkdir(parents=True, exist_ok=True)
                (workspace / "memory").mkdir(exist_ok=True)
                (workspace / "tools").mkdir(exist_ok=True)

                agent = Agent(agent_config, session_id=str(channel_id))
                self._agents[channel_id] = agent
                logger.info(
                    f"Loaded agent '{agent_name}' for #{ch_cfg.get('name', channel_id)}"
                )

            except Exception as e:
                logger.exception(f"Failed to create agent for channel {channel_id}")
                return None

        return self._agents[channel_id]

    async def _send_response(
        self,
        channel,
        text: str,
        reply_to: Message = None,
    ):
        """
        Send a response, splitting into multiple messages if needed.
        First chunk replies to the original message; subsequent chunks
        are sent as follow-up messages.
        """
        if not text:
            return

        chunks = self._chunk_message(text)

        for i, chunk in enumerate(chunks):
            try:
                if i == 0 and reply_to:
                    await reply_to.reply(chunk, mention_author=False)
                else:
                    await channel.send(chunk)
            except discord.HTTPException as e:
                if e.status == 429:
                    # Rate limited
                    retry_after = getattr(e, "retry_after", 5.0)
                    logger.warning(f"Rate limited, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    try:
                        if i == 0 and reply_to:
                            await reply_to.reply(chunk, mention_author=False)
                        else:
                            await channel.send(chunk)
                    except discord.HTTPException:
                        logger.error(f"Failed to send chunk {i} after retry")
                else:
                    logger.error(f"Failed to send message: {e}")

    @staticmethod
    def _chunk_message(text: str, limit: int = MAX_MESSAGE_LENGTH) -> list[str]:
        """
        Split a long message into chunks that fit Discord's limit.

        Split priority: paragraph breaks -> newlines -> spaces -> hard cut.
        """
        if len(text) <= limit:
            return [text]

        chunks = []
        remaining = text

        while remaining:
            if len(remaining) <= limit:
                chunks.append(remaining)
                break

            # Find a good split point
            split_at = limit

            # Try double newline (paragraph break)
            idx = remaining.rfind("\n\n", 0, limit)
            if idx > limit // 2:
                split_at = idx + 2
            else:
                # Try single newline
                idx = remaining.rfind("\n", 0, limit)
                if idx > limit // 2:
                    split_at = idx + 1
                else:
                    # Try space
                    idx = remaining.rfind(" ", 0, limit)
                    if idx > limit // 2:
                        split_at = idx + 1

            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:]

        return chunks

    def run(self, foreground: bool = True):
        """
        Start the Discord bot. Blocks until disconnected (Ctrl+C).

        Args:
            foreground: If True, runs interactively with console output.
                       If False, runs quietly (for daemon mode — logging goes to file).
        """
        token = self._discord_cfg["bot_token"]

        if foreground:
            # Interactive console logging
            logging.basicConfig(
                level=logging.INFO,
                format="  [discord] %(message)s",
            )
            _quiet_noisy_loggers()

            print("  Starting Alfred Discord bot...")
            print(f"  Watching {len(self._channel_configs)} channel(s)")
            print("  Press Ctrl+C to stop.\n")
        else:
            # File logging for daemon mode
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
                filename=str(LOG_FILE),
                filemode="a",
            )
            _quiet_noisy_loggers()

        try:
            self._client.run(token, log_handler=None)
        except KeyboardInterrupt:
            if foreground:
                print("\n  Shutting down...")
        finally:
            PID_FILE.unlink(missing_ok=True)

    # Daemon mode is handled by cmd_start() in __main__.py using subprocess.Popen.
    # This avoids fork-safety issues with LanceDB.
