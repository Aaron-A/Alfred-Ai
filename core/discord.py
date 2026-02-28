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
import time
from typing import Optional
from pathlib import Path
from collections import defaultdict

import discord
from discord import Message, Thread

from .config import _load_config, config
from .agent import Agent, AgentConfig
from .logging import get_logger, setup_logging

logger = get_logger("discord")

# Rate limit defaults
DEFAULT_RATE_LIMIT_MESSAGES = 5      # messages per window
DEFAULT_RATE_LIMIT_WINDOW = 60       # window in seconds
DEFAULT_RATE_LIMIT_COOLDOWN = 30     # cooldown after hitting limit

# PID and log file locations
PID_FILE = config.PROJECT_ROOT / "data" / "discord.pid"
LOG_FILE = config.PROJECT_ROOT / "data" / "discord.log"
HEARTBEAT_FILE = config.PROJECT_ROOT / "data" / "heartbeat"

MAX_MESSAGE_LENGTH = 2000  # Discord's hard limit
HEARTBEAT_INTERVAL = 30  # seconds


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


def write_heartbeat():
    """Write a heartbeat timestamp file (called periodically by the daemon)."""
    try:
        HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT_FILE.write_text(f"{time.time()}\n{os.getpid()}")
    except Exception:
        pass


def read_heartbeat() -> Optional[tuple[float, int]]:
    """Read the heartbeat file. Returns (timestamp, pid) or None."""
    if not HEARTBEAT_FILE.exists():
        return None
    try:
        parts = HEARTBEAT_FILE.read_text().strip().split("\n")
        return float(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return None


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
        HEARTBEAT_FILE.unlink(missing_ok=True)
        return None


def is_bot_healthy() -> tuple[bool, str]:
    """
    Check if the bot is running AND healthy (heartbeat within last 90s).

    Returns:
        (healthy, status_message)
    """
    pid = is_bot_running()
    if pid is None:
        return False, "not running"

    heartbeat = read_heartbeat()
    if heartbeat is None:
        # Process alive but no heartbeat — still starting up or heartbeat not implemented
        return True, f"running (PID {pid}, no heartbeat yet)"

    ts, hb_pid = heartbeat
    age = time.time() - ts

    if hb_pid != pid:
        return True, f"running (PID {pid}, stale heartbeat from PID {hb_pid})"

    if age > 90:
        return False, f"unresponsive (PID {pid}, last heartbeat {int(age)}s ago)"

    return True, f"healthy (PID {pid}, heartbeat {int(age)}s ago)"


def _kill_and_wait(pid: int, timeout: float = 5.0) -> bool:
    """Send SIGTERM, wait up to *timeout* seconds, escalate to SIGKILL.

    Returns True if the process is confirmed dead.
    """
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.1)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True

    # Still alive — force kill
    try:
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.3)
    except ProcessLookupError:
        pass
    return True


def _find_daemon_pids() -> list[int]:
    """Find all running Alfred daemon-child processes by scanning /proc (macOS: ps).

    Returns a list of PIDs (may be empty).  Never includes the current process.
    """
    import subprocess as _sp
    my_pid = os.getpid()
    try:
        out = _sp.check_output(
            ["pgrep", "-f", "__main__.py start --daemon-child"],
            text=True,
        )
        return [int(p) for p in out.strip().split("\n") if p.strip() and int(p) != my_pid]
    except (_sp.CalledProcessError, FileNotFoundError, ValueError):
        return []


def stop_bot() -> bool:
    """Stop all running Alfred daemon processes.

    1. Kill the PID recorded in the PID file (primary).
    2. Scan for any other daemon-child processes that may have been orphaned
       (e.g. during a reload race) and kill those too.
    3. Clean up PID and heartbeat files.
    """
    pid = is_bot_running()
    killed_any = False

    # Kill the primary PID from the file
    if pid is not None:
        killed_any = _kill_and_wait(pid)

    # Scan for any orphaned daemon-child processes the PID file missed
    orphans = _find_daemon_pids()
    for opid in orphans:
        if opid != pid:  # don't double-kill
            _kill_and_wait(opid)
            killed_any = True

    # Clean up
    PID_FILE.unlink(missing_ok=True)
    HEARTBEAT_FILE.unlink(missing_ok=True)

    return killed_any


class RateLimiter:
    """
    Per-user rate limiter using a sliding window.

    Tracks message timestamps per user. If a user sends more than
    `max_messages` within `window_seconds`, they're rate-limited for
    `cooldown_seconds`.

    Config in alfred.json under discord.rate_limit:
        {
            "messages_per_minute": 5,    # max messages in the window
            "window_seconds": 60,        # sliding window size
            "cooldown_seconds": 30       # cooldown after hitting limit
        }
    """

    def __init__(
        self,
        max_messages: int = DEFAULT_RATE_LIMIT_MESSAGES,
        window_seconds: int = DEFAULT_RATE_LIMIT_WINDOW,
        cooldown_seconds: int = DEFAULT_RATE_LIMIT_COOLDOWN,
    ):
        self.max_messages = max_messages
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds

        # user_id -> list of timestamps (within the current window)
        self._timestamps: dict[int, list[float]] = defaultdict(list)

        # user_id -> cooldown expiry time (when rate-limited)
        self._cooldowns: dict[int, float] = {}

    def check(self, user_id: int) -> tuple[bool, str]:
        """
        Check if a user is allowed to send a message.

        Returns:
            (allowed, reason) — True if allowed, False with explanation if not
        """
        now = time.monotonic()

        # Check if user is in cooldown
        if user_id in self._cooldowns:
            expiry = self._cooldowns[user_id]
            if now < expiry:
                remaining = int(expiry - now)
                return False, f"Rate limited — try again in {remaining}s"
            else:
                # Cooldown expired, remove it
                del self._cooldowns[user_id]

        # Clean old timestamps outside the window
        cutoff = now - self.window_seconds
        self._timestamps[user_id] = [
            ts for ts in self._timestamps[user_id] if ts > cutoff
        ]

        # Check if under the limit
        if len(self._timestamps[user_id]) >= self.max_messages:
            # Rate limit triggered — start cooldown
            self._cooldowns[user_id] = now + self.cooldown_seconds
            logger.warning(
                f"Rate limited user {user_id}: "
                f"{self.max_messages} msgs in {self.window_seconds}s, "
                f"cooldown {self.cooldown_seconds}s"
            )
            return False, (
                f"Slow down! Max {self.max_messages} messages per "
                f"{self.window_seconds}s. Try again in {self.cooldown_seconds}s."
            )

        # Record this message
        self._timestamps[user_id].append(now)
        return True, ""

    @classmethod
    def from_config(cls, discord_cfg: dict) -> "RateLimiter":
        """Create a RateLimiter from the discord config block."""
        rl_cfg = discord_cfg.get("rate_limit", {})
        return cls(
            max_messages=rl_cfg.get("messages_per_minute", DEFAULT_RATE_LIMIT_MESSAGES),
            window_seconds=rl_cfg.get("window_seconds", DEFAULT_RATE_LIMIT_WINDOW),
            cooldown_seconds=rl_cfg.get("cooldown_seconds", DEFAULT_RATE_LIMIT_COOLDOWN),
        )


class DiscordBot:
    """
    Discord bot that routes channel messages to Alfred agents.

    One Agent instance per channel for thread safety.
    Thread messages inherit their parent channel's agent.
    Rate-limited per user to prevent API credit abuse.
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

        # Per-user rate limiter
        self._rate_limiter = RateLimiter.from_config(self._discord_cfg)

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

            # Log rate limit config
            rl = self._rate_limiter
            logger.info(
                f"  Rate limit: {rl.max_messages} msgs / {rl.window_seconds}s, "
                f"{rl.cooldown_seconds}s cooldown"
            )

            # Set presence
            activity = discord.Activity(
                type=discord.ActivityType.listening,
                name=f"{len(self._channel_configs)} channels",
            )
            await self._client.change_presence(activity=activity)

            # Start heartbeat loop (for daemon health checks)
            self._client.loop.create_task(self._heartbeat_loop())

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

        # 5. Rate limit check (per user)
        allowed, reason = self._rate_limiter.check(message.author.id)
        if not allowed:
            try:
                await message.add_reaction("⏳")
                await message.reply(reason, mention_author=True, delete_after=15)
            except discord.HTTPException:
                pass
            return

        # 6. Extract clean message text (strip @mention)
        text = self._clean_message_text(message)
        if not text.strip():
            return

        # 7. Get per-channel lock
        if effective_channel_id not in self._channel_locks:
            self._channel_locks[effective_channel_id] = asyncio.Lock()

        async with self._channel_locks[effective_channel_id]:
            # 8. Get or create agent for this channel
            agent = self._get_agent(effective_channel_id)
            if agent is None:
                logger.error(f"No agent for channel {effective_channel_id}")
                return

            # 9. Check streaming preference (per-channel or global)
            use_streaming = ch_cfg.get("stream", self._discord_cfg.get("stream", False))

            try:
                if use_streaming:
                    await self._handle_streaming(message, agent, text)
                else:
                    await self._handle_blocking(message, agent, text)

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

    async def _handle_blocking(self, message: Message, agent: Agent, text: str):
        """Standard blocking response — wait for full response, then send."""
        async with message.channel.typing():
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, agent.run, text)

        await self._send_response(message.channel, response, reply_to=message)

    async def _handle_streaming(self, message: Message, agent: Agent, text: str):
        """
        Progressive streaming — edit message as tokens arrive.

        Creates a reply immediately with "..." then edits it every ~1s
        with accumulated text. Final edit contains the complete response.
        If the response exceeds Discord's limit, sends additional messages.
        """
        import queue as queue_mod

        q = queue_mod.Queue()
        loop = asyncio.get_event_loop()

        def _stream_worker():
            try:
                for chunk in agent.run_stream(text):
                    q.put(("chunk", chunk))
                q.put(("done", None))
            except Exception as e:
                q.put(("error", str(e)))

        # Start streaming in background thread
        fut = loop.run_in_executor(None, _stream_worker)

        # Send initial reply
        try:
            bot_msg = await message.reply("...", mention_author=False)
        except discord.HTTPException:
            bot_msg = await message.channel.send("...")

        accumulated = ""
        last_edit_time = time.monotonic()
        edit_interval = 1.0  # Edit at most once per second
        done = False

        while not done:
            # Drain the queue
            try:
                while True:
                    msg_type, data = q.get_nowait()
                    if msg_type == "chunk":
                        accumulated += data
                    elif msg_type == "done":
                        done = True
                        break
                    elif msg_type == "error":
                        accumulated += f"\n\n*Error: {data[:200]}*"
                        done = True
                        break
            except queue_mod.Empty:
                pass

            # Edit the message if enough time has passed or we're done
            now = time.monotonic()
            if accumulated and (done or now - last_edit_time >= edit_interval):
                display = accumulated[:MAX_MESSAGE_LENGTH]
                if not done and len(accumulated) <= MAX_MESSAGE_LENGTH:
                    display += " ..."
                try:
                    await bot_msg.edit(content=display)
                    last_edit_time = now
                except discord.HTTPException:
                    pass

            if not done:
                await asyncio.sleep(0.1)

        # Final: if response exceeded Discord limit, send overflow as new messages
        if len(accumulated) > MAX_MESSAGE_LENGTH:
            # Edit first message to show first chunk
            try:
                await bot_msg.edit(content=accumulated[:MAX_MESSAGE_LENGTH])
            except discord.HTTPException:
                pass
            # Send remaining chunks
            remaining = accumulated[MAX_MESSAGE_LENGTH:]
            overflow_chunks = self._chunk_message(remaining)
            for chunk in overflow_chunks:
                try:
                    await message.channel.send(chunk)
                except discord.HTTPException:
                    break

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

    async def _heartbeat_loop(self):
        """Write heartbeat file every HEARTBEAT_INTERVAL seconds."""
        while not self._client.is_closed():
            write_heartbeat()
            await asyncio.sleep(HEARTBEAT_INTERVAL)

    def post_to_agent_channel(self, agent_name: str, text: str) -> bool:
        """Post a message to the Discord channel mapped to an agent.

        Thread-safe — can be called from scheduler threads. Bridges into
        discord.py's async event loop via run_coroutine_threadsafe.
        """
        if not text:
            return False

        # Find the channel mapped to this agent
        channel_id = None
        for ch_id, ch_cfg in self._channel_configs.items():
            if ch_cfg.get("agent") == agent_name:
                channel_id = ch_id
                break

        if channel_id is None:
            logger.debug(f"No Discord channel mapped to agent '{agent_name}'")
            return False

        loop = getattr(self._client, "loop", None)
        if not loop or not loop.is_running():
            logger.debug("Discord event loop not running, skipping post")
            return False

        channel = self._client.get_channel(channel_id)
        if not channel:
            logger.warning(f"Discord channel {channel_id} not found (bot may not be ready)")
            return False

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._send_response(channel, text), loop
            )
            future.result(timeout=30)
            return True
        except Exception as e:
            logger.warning(f"Failed to post to Discord channel {channel_id}: {e}")
            return False

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
            setup_logging(to_console=True, to_file=False)

            print("  Starting Alfred Discord bot...")
            print(f"  Watching {len(self._channel_configs)} channel(s)")
            print("  Press Ctrl+C to stop.\n")
        else:
            # File logging for daemon mode
            from .logging import LOG_FILE as _log_file
            _log_file.parent.mkdir(parents=True, exist_ok=True)
            setup_logging(to_console=False, to_file=True)

        # Register signal handlers for graceful shutdown
        def _signal_handler(signum, frame):
            signame = signal.Signals(signum).name
            logger.info(f"Received {signame}, shutting down...")
            # Schedule the close on the event loop
            if self._client.loop and self._client.loop.is_running():
                self._client.loop.create_task(self._client.close())
            else:
                raise SystemExit(0)

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)

        # Write initial heartbeat
        write_heartbeat()

        try:
            self._client.run(token, log_handler=None)
        except (KeyboardInterrupt, SystemExit):
            if foreground:
                print("\n  Shutting down...")
        finally:
            # Only clean up PID file if it still belongs to us (avoid clobbering
            # a new daemon's PID during reload restarts)
            try:
                if PID_FILE.exists() and int(PID_FILE.read_text().strip()) == os.getpid():
                    PID_FILE.unlink(missing_ok=True)
            except (ValueError, OSError):
                PID_FILE.unlink(missing_ok=True)
            HEARTBEAT_FILE.unlink(missing_ok=True)
            logger.info("Alfred stopped")

    # Daemon mode is handled by cmd_start() in __main__.py using subprocess.Popen.
    # This avoids fork-safety issues with LanceDB.
