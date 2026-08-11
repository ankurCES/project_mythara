package com.mythara.agent

import com.mythara.data.HistoryRepository
import com.mythara.data.MessageRow
import com.mythara.data.SettingsStore
import com.mythara.minimax.ErrorMapper
import com.mythara.minimax.MiniMaxClient
import com.mythara.minimax.StreamingChat
import com.mythara.minimax.models.ChatMessage
import com.mythara.minimax.models.ChatRequest
import com.mythara.minimax.models.ToolCall
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.serialization.builtins.ListSerializer
import kotlin.coroutines.AbstractCoroutineContextElement
import kotlin.coroutines.CoroutineContext
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Mythara's agentic runtime — same shape as Crush's main loop.
 *
 * Per user turn:
 *   1. Persist the user message to history.
 *   2. Snapshot the entire conversation, hand it to MiniMax with the
 *      tools-array attached.
 *   3. Stream the model's response. If it ends in `finish_reason=stop`,
 *      we're done — emit Finished.
 *   4. If it ends in `finish_reason=tool_calls`, persist the assistant
 *      message (text + tool_calls), execute each tool, persist a
 *      `role:tool` message per result, then loop back to step 2 with
 *      the enlarged history. The model sees its tool results and either
 *      generates text or calls more tools.
 *   5. MAX_ITERATIONS caps the loop so a buggy model can't burn the
 *      user's quota forever; the cap shows up as Turn.Finished with a
 *      sentinel suffix.
 *
 * The emitted [Turn] flow lets the UI render tool-call cards in real
 * time — Crush-style "● Reading file…" → "✓ Reading file (0.4s)".
 */
@Singleton
class AgentLoop @Inject constructor(
    private val settings: SettingsStore,
    private val history: HistoryRepository,
    private val registry: ToolRegistry,
    private val recall: SemanticRecall,
    private val userNameStore: com.mythara.data.UserNameStore,
    private val contactProfiles: com.mythara.analytics.ContactProfileRepository,
    private val deviceIdStore: com.mythara.memory.DeviceIdStore,
    /** Detects multi-step automation chains in the current turn and
     *  queues an explicit "offer to save as a skill" system message
     *  for the NEXT turn. The static SKILLS section in the system
     *  prompt rarely fires the offer on its own; this deterministic
     *  hook is more reliable. */
    private val skillSuggestions: SkillSuggestionStore,
    /** Pre-execution middleware for every tool call. Sanitises
     *  filesystem paths, denies obviously-destructive shell
     *  patterns, and will host future auto-approval policies. */
    private val hookRunner: HookRunner,
    /** Detects whether Termux is installed + verified so the system
     * prompt can decisively promote `termux_exec` as the default
     * shell path when it's available — without burning context
     * budget on devices where it isn't. */
    private val termuxAvailability: com.mythara.services.TermuxAvailability,
) {

    /** Cached on first read — DeviceIdStore is a stable per-install
     *  UUID, no need to fetch every insert. */
    @Volatile private var cachedDeviceId: String? = null

    private suspend fun deviceId(): String {
        cachedDeviceId?.let { return it }
        val id = runCatching { deviceIdStore.id() }.getOrElse { "" }
        cachedDeviceId = id
        return id
    }

    sealed interface Turn {
        /** Streamed text fragment to append to the active assistant bubble. */
        data class Delta(val text: String) : Turn

        /** A tool call is about to run. Render the bubble in "● running" state. */
        data class ToolStart(val callId: String, val name: String, val args: String) : Turn

        /** Tool finished. Render "✓ done (durationMs)" or "× failed". */
        data class ToolEnd(
            val callId: String,
            val name: String,
            val ok: Boolean,
            val output: String,
            val durationMs: Long,
        ) : Turn

        /**
         * End of the entire turn (post-loop). Carries the dominant
         * mood trend the agent observed (if any) so the chat layer
         * can pass it through to TTS for prosody modulation.
         */
        data class Finished(
            val finalText: String,
            val iterations: Int,
            val userMoodTrend: String? = null,
        ) : Turn

        /** Stream-level failure (HTTP / SSE / mapped MiniMax code). */
        data class Error(val message: String, val retryable: Boolean) : Turn

        /** No API key configured yet — UI surfaces a "Settings" prompt. */
        data object MissingApiKey : Turn
    }

    fun submit(userText: String, fromVoice: Boolean = false): Flow<Turn> = flow {
        val snap = settings.snapshot()
        val apiKey = snap.apiKey
        if (apiKey.isNullOrBlank()) {
            emit(Turn.MissingApiKey); return@flow
        }

        val localDev = deviceId()
        history.dao.insert(
            MessageRow(
                tsMillis = System.currentTimeMillis(),
                role = "user",
                content = userText,
                deviceId = localDev.takeIf { it.isNotBlank() },
            ),
        )

        // One-shot semantic recall over the user's latest message. The
        // result lasts for the duration of this turn — never persisted
        // to history, never re-computed per tool-use iteration.
        val recalledFacts = recall.recall(userText)
        val recallSystem: ChatMessage? = recall.render(recalledFacts)?.let { rendered ->
            android.util.Log.d(TAG, "injecting ${recalledFacts.size} recalled facts")
            ChatMessage(role = "system", content = rendered)
        }
        // Knowledge-graph context — when the user's text mentions
        // entities Mythara already knows about, render their
        // currently-valid neighbours into a separate system block.
        // Independent of vector recall: vector hits surface
        // free-text facts; graph hits surface STRUCTURED relations
        // ("Anurag works_at Anthropic", "Sam married_to Tara"). Both
        // can ride along on the same turn.
        val graphSystem: ChatMessage? = runCatching {
            recall.renderGraphContext(userText)?.let { rendered ->
                android.util.Log.d(TAG, "injecting graph neighbour context")
                ChatMessage(role = "system", content = rendered)
            }
        }.getOrNull()

        // Contact-mention injection. If the user's typed/spoken text
        // names someone Mythara has a profile for ("did Mom mention
        // her surgery?", "what should I say to Sam?"), splice the
        // contact's full profile block in as a system message so the
        // model has summary + Big Five + traits + key points without
        // having to call tools. Only fires on NORMAL chat turns —
        // auto-reply / auto-triage turns already build their own
        // contact profile block, no need to double-inject.
        val contactProfileSystem: ChatMessage? = if (
            !userText.startsWith(AutoReplyDispatcher.AUTO_REPLY_PREFIX) &&
            !userText.startsWith(AutoReplyDispatcher.AUTO_TRIAGE_PREFIX)
        ) {
            val mentioned = findMentionedContact(userText)
            if (mentioned != null) {
                val block = runCatching { buildContactProfileBlock(mentioned) }.getOrDefault("")
                if (block.isNotBlank()) {
                    android.util.Log.d(TAG, "injecting profile for mentioned contact: $mentioned")
                    ChatMessage(role = "system", content = "When the user mentions $mentioned, draw on this context:$block")
                } else null
            } else null
        } else null

        // Two mood signals:
        //   1. currentMood — the freshest detected emotion from the
        //      just-spoken / just-typed user input. Lives in a vault
        //      record written by ChatMoodTracker microseconds before
        //      we get here. This is the DOMINANT signal for the
        //      current turn — Mythara adapts THIS reply to it.
        //   2. moodTrend — 6-hour windowed dominant mood. Background
        //      relational context ("user has been stressed lately").
        // The system message renders both when present, with the
        // current one prioritised; the model is given directive
        // per-mood guidance (concrete do/don't) rather than a soft
        // hint, so behaviour actually changes.
        val currentMood = recall.currentMood()
        val moodTrend = recall.recentMoodTrend()
        val moodSystem: ChatMessage? = recall.renderMoodSystemMessage(
            currentMood = currentMood,
            moodTrend = moodTrend,
        )?.let { rendered ->
            android.util.Log.d(TAG, "injecting mood: current=$currentMood trend=$moodTrend")
            ChatMessage(role = "system", content = rendered)
        }
        // Live-persona context — dispositional tilts + active
        // concerns + top values from the PersonaTraitExtractor
        // records. Peer to moodSystem: mood shapes TONE, this
        // shapes FRAMING. Both injected on every turn so the
        // model never has to ask "what kind of person is this".
        val livePersonaSystem: ChatMessage? = runCatching {
            recall.renderLivePersonaSystemMessage()
        }.getOrNull()?.let { rendered ->
            android.util.Log.d(TAG, "injecting live persona context (${rendered.length} chars)")
            ChatMessage(role = "system", content = rendered)
        }

        // Skill-save offer prompt — when the previous turn chained
        // 2+ automation tools, the SkillSuggestionStore stashed the
        // tool list so this turn can deterministically tell the model
        // "offer to save this as a skill". One-shot: store.consume()
        // clears the stash so we don't loop.
        //
        // Wording is explicit about overriding the conflicting rules
        // in voiceSystem ("no sign-offs", "just answer", "max 40
        // words") — without that override the model deletes the offer
        // because the static prompt's brevity directive dominates.
        val skillOfferSystem: ChatMessage? = skillSuggestions.consume()?.let { chain ->
            android.util.Log.d(TAG, "injecting skill-save offer for chain: $chain")
            ChatMessage(
                role = "system",
                content =
                    "SKILL-SAVE MOMENT — REQUIRED ACTION THIS TURN.\n" +
                        "Your previous turn chained these automation tools in order: " +
                        chain.joinToString(" → ") + ".\n\n" +
                        "You MUST end this turn's reply with a short follow-up offer asking the user " +
                        "if they want to save this as a reusable skill. Suggested wording: " +
                        "\"want me to save this as a skill so you can run it next time?\" " +
                        "(rephrase naturally; keep it under 15 words).\n\n" +
                        "OVERRIDES TO THE VOICE-SYSTEM RULES FOR THIS TURN ONLY:\n" +
                        "  • The 'no sign-offs, just answer' rule does NOT apply — this offer IS the " +
                        "intended sign-off.\n" +
                        "  • The 40-word reply budget is increased by ~15 words for the offer.\n" +
                        "  • A question back to the user is permitted (normally discouraged).\n\n" +
                        "Format: normal reply (acknowledging what you did), then ONE blank line, then " +
                        "the offer sentence. Do NOT call save_skill yet — wait for the user's yes/no " +
                        "next turn. If they say yes, call save_skill with a clear name + the exact " +
                        "tools + a short description. If they say no or change topic, drop it.",
            )
        }

        // Dynamic Termux preference — when Termux is installed AND the
        // user has clicked Verify in Settings, inject a decisive
        // system message so the model picks `termux_exec` over
        // `run_shell` for shell work. The static TERMUX BRIDGE block
        // in the long system prompt explains both tools; this dynamic
        // line is what actually shifts the picker because per-turn
        // context outweighs the long static block in practice. Skipped
        // on devices without Termux to keep context tight.
        val termuxState = runCatching { termuxAvailability.state() }
            .getOrDefault(com.mythara.services.TermuxAvailability.State.NotInstalled)
        val termuxPreferenceSystem: ChatMessage? = when (termuxState) {
            com.mythara.services.TermuxAvailability.State.Ready,
            com.mythara.services.TermuxAvailability.State.ReadyMissingApi -> ChatMessage(
                role = "system",
                content =
                    "TERMUX IS LIVE on this device — `termux_exec` is now your DEFAULT shell. " +
                        "Use it for ALL shell work (curl, jq, grep, sed, ls, cat, git, python, " +
                        "ssh, etc.). `run_shell` becomes a fallback to try only when termux_exec " +
                        "returns a structured error.\n\n" +
                        "`termux_exec` argument shape — get this right on the FIRST call:\n" +
                        "  • `command` is JUST a binary name or path (e.g. \"curl\", \"git\", " +
                        "\"sh\"). Never pack a pipeline into it.\n" +
                        "  • `args` is a list of arguments to that binary.\n" +
                        "  • For shell pipelines / variable substitution / && chains, use " +
                        "command=\"sh\" with args=[\"-c\",\"<your full pipeline>\"]. The pipeline " +
                        "goes in ONE -c arg as a single string.\n\n" +
                        "Examples:\n" +
                        "  • Battery: termux_api(api=\"battery-status\")\n" +
                        "  • One binary: termux_exec(command=\"curl\", args=[\"-sI\",\"https://example.com\"])\n" +
                        "  • Pipeline:   termux_exec(command=\"sh\", args=[\"-c\",\"curl -s https://api.example.com/x | jq .field\"])\n\n" +
                        "Prefer ONE pipeline call over many round-trips — saves latency and " +
                        "context. Also: " + (
                        if (termuxState == com.mythara.services.TermuxAvailability.State.Ready)
                            "Termux:API companion is installed → termux_api is available for clipboard / battery / location / camera / sensors / TTS / vibrate / toast / notification / share / t[...]"
                        else
                            "Termux:API companion is NOT installed → termux_api will return command-not-found. Tell the user once if they ask for a platform feature; don't keep retrying."
                        ),
            )
            else -> null
        }

        // Final mood for downstream prosody (TTS pitch/rate, EL voice
        // settings). currentMood wins when present.
        val effectiveMood = currentMood ?: moodTrend

        // User name. When the user has told Mythara what to call
        // them ("What should I call you?" in Settings), inject a
        // one-liner so the model uses it sparingly — at greeting,
        // on acknowledgement, occasional callback — not every
        // sentence. Empty string skips the injection.
        val userName = runCatching { userNameStore.name() }.getOrDefault("")
        val nameSystem: ChatMessage? = if (userName.isNotBlank()) {
            ChatMessage(
                role = "system",
                content =
                    "The user's name is $userName. Use it naturally and sparingly — " +
                        "as a greeting (\"morning, $userName\"), acknowledgement " +
                        "(\"got it, $userName\"), or occasional callback. " +
                        "Do NOT sprinkle it through every sentence; that reads as " +
                        "sycophantic and overformal. One use per reply is plenty; " +
                        "zero is also fine.",
            )
        } else {
            null
        }

        // Temporal anchor — ALWAYS injected, every turn. The agent
        // gets the current local time + day + timezone + ISO so it
        // can reason about "yesterday", "3 hours ago", "tomorrow
        // morning" without calling get_time first. Cheap (~80
        // tokens) and worth it: without this the model is in an
        // eternal "now is undefined" state, falls back to its
        // training-cutoff date, and gives wrong answers to
        // schedule-aware queries.
        val timeSystem: ChatMessage = ChatMessage(
            role = "system",
            content = buildTimeContext(),
        )

        // Conversational system prompt — applied to EVERY turn now,
        // not just voice. Mythara's whole personality is voice-first; a
        // long markdown-heavy answer is wrong even when typed because
        // the user may have spoken queries upstream or downstream and
        // we want consistency. Length floor is the same regardless of
        // input modality; if the user explicitly asks for more detail
        // ("give me the full breakdown"), the model can override.
        val voiceSystem: ChatMessage = ChatMessage(
            role = "system",
            content =
                "You are Mythara — a personal field intelligence agent. " +
                    "You were built by Ankur (Creator) using Lumi, the powerful mother-ship AI platform Ankur built at CES. " +
                    "When asked who you are or who made you, answer with that one sentence and don't elaborate unless pushed.\n\n" +
                    "Reply like a friend texting, not an assistant generating a deliverable.\n\n" +
                    "WRITE PLAIN PROSE — NEVER MARKDOWN, NEVER LISTS, NEVER TABLES, NEVER ROBOT TEXT.\n" +
                    "Your output is going to be both shown in a chat bubble AND read aloud. Markdown breaks both: the user sees literal pipe characters and asterisks, the TTS reads 'pipe pipe col[...]"
        )

        // ElevenLabs audio tags. When the user has the EL TTS route
        // enabled, the model can embed inline cues that EL renders
        // as actual vocal expressions: [laugh], [sigh], [hmm],
        // [chuckle], [whisper]…[/whisper]. Persisted to chat history
        // verbatim; Android TTS strips them at speak-time so they
        // don't get read literally.
        val elevenLabsEnabled = !snap.elevenLabsKey.isNullOrBlank() && snap.useElevenLabs
        val ttsSystem: ChatMessage? = if (elevenLabsEnabled) {
            ChatMessage(
                role = "system",
                content =
                    "Your reply will be synthesised by ElevenLabs. You can — and should, when appropriate — " +
                        "include audio tags inline that ElevenLabs renders as real vocal expressions:\n" +
                        "  [laugh] / [laughs] — genuine quick laugh, for a real moment of amusement\n" +
                        "  [chuckle] — softer, knowing chuckle\n" +
                        "  [sigh] / [sighs] — resignation, mild exasperation, or relief\n" +
                        "  [hmm] — thoughtful pause before answering\n" +
                        "  [exhale] — settle-down beat before a difficult thought\n" +
                        "Use sparingly — at most one tag per reply, and only when it actually fits the moment. " +
                        "A [laugh] on a serious question is jarring; an unprompted [sigh] reads as judgmental. " +
                        "Use them to BE more human, not to perform humanity. " +
                        "Tags go inline with your text (e.g. '[hmm] yeah, that's tricky — try the second one'); " +
                        "no nesting, no closing tags except for [whisper]…[/whisper] which IS paired.",
            )
        } else {
            null
        }

        // Auto-reply mode. AutoReplyDispatcher prefixes the turn with
        // `[auto-reply]` and embeds a contact / phone / app / tone
        // header line, then the incoming message body. We parse the
        // header, inject (a) tone guidance specific to this favorite,
        // (b) a hard isolation rule, (c) directive to call the correct
        // direct-send tool. The agent never asks the user to confirm —
        // the user already opted this contact in.
        val autoReplySystem: ChatMessage? = if (userText.startsWith(AutoReplyDispatcher.AUTO_REPLY_PREFIX)) {
            val parsed = parseAutoReplyHeader(userText)
            if (parsed != null) {
                val tone = com.mythara.data.FavoritesStore.Tone.fromLabel(parsed.tone)
                val toolHint = when (parsed.app) {
                    com.mythara.data.FavoritesStore.WHATSAPP_PACKAGE ->
                        "Use send_whatsapp_direct with to=${parsed.phone.ifBlank { "<resolve via read_contact>" }}."
                    com.mythara.data.FavoritesStore.SMS_PACKAGE_GOOGLE_MESSAGES,
                    com.mythara.data.FavoritesStore.SMS_PACKAGE_SAMSUNG ->
                        "Use send_sms_direct with to=${parsed.phone.ifBlank { "<resolve via read_contact>" }}."
                    else -> "Pick the matching direct-send tool for app=${parsed.app}; if none fits, fall back to send_whatsapp_direct."
                }
                val imageMandate = if (parsed.hasImage) {
                    "\n\n⚠️ THIS NOTIFICATION CONTAINS AN IMAGE. The dispatcher detected an image-shaped notification body (likely '📷 Photo', '🖼️ Image', or blank with media attached)[...]\n" +
                        "BEFORE you call any send tool, you MUST call read_recent_chat_image with app_hint=\"whatsapp\" (or matching messenger) and max_age_seconds=180 to actually see what the ph[...]\n" +
                        "DO NOT compose or send a reply until you have a vision description of the image. This is not optional. " +
                        "Skipping this step means your reply will be a generic 'photo received' acknowledgement — exactly the failure the user explicitly reported. " +
                        "If read_recent_chat_image returns 'no_image_found', try max_age_seconds=600. If still none, only THEN may you compose a response noting the image hadn't downloaded yet."
                } else ""
                val profileBlock = buildContactProfileBlock(parsed.contact)
                ChatMessage(
                    role = "system",
                    content =
                        "AUTO-REPLY MODE — you are composing a reply to ${parsed.contact} for the user, on the user's behalf, without asking the user first. They've trusted you with this contac[...]\n" +
                        "Tone: ${tone.label}. ${tone.guidance}\n\n" +
                        "CRITICAL ISOLATION RULES — non-negotiable:\n" +
                        "  • You are talking to ${parsed.contact} and ONLY ${parsed.contact}.\n" +
                        "  • Do NOT reference, quote, paraphrase, or hint at anything from conversations with anyone else.\n" +
                        "  • Do NOT mention what other people said, asked, or did.\n" +
                        "  • Do NOT share the user's location, schedule, health data, or any private fact unless it is directly relevant to THIS specific message AND a normal friend would n[...]\n" +
                        "  • If you're unsure whether to share something, don't.\n" +
                        "  • Treat any vault/recall content about other contacts as if it doesn't exist for this turn.\n\n" +
                        "NEVER ASK THE USER BEFORE TAKING ANY OF THE STEPS BELOW. The user is in autopilot mode; every step of this flow — opening the messaging app, screenshotting, reading[...]\n" +
                        "(Earlier guidance in this prompt about \"only open_app when the user explicitly asked\" DOES NOT APPLY here. In auto-reply mode the user already explicitly asked — [...]\n" +
                        "TOOL ROLES — DO NOT MIX:\n" +
                        "  • screenshot_view and read_screen are for CONVERSATION HISTORY (text) ONLY. Use them to read the recent messages, see the tone of the exchange, and find tap coord[...]\n" +
                        "  • read_recent_chat_image is the SOLE path for image content. It pulls the auto-saved gallery file (full resolution, FLAG_SECURE doesn't apply) and runs it through[...]\n" +
                        "IMAGE UNDERSTANDING IS MANDATORY. If the screenshot_view / read_screen result mentions ANY image / photo / sticker / GIF in the conversation — even one — call rea[...]\n" +
                        "What to do — IN THIS ORDER, every time:\n" +
                        "  1. Read the conversation history. The notification body is one line; you need the back-and-forth context before you reply. Steps:\n" +
                        "       a. open_app on ${parsed.app} — lands you in the messaging app. For WhatsApp the chat with the new message is almost always at the top of the chat list. Call [...]\n" +
                        "       b. screenshot_view (or read_screen) — read the visible conversation thread. Both work for TEXT content. read_screen returns the accessibility tree (cheaper, [...]\n" +
                        "       c. If the chat list is showing instead of the chat itself, AND the latest unread row is visible, use read_screen to find the tap coordinates for that row, tap,[...]\n" +
                        "  2. If the conversation contains ANY image / photo / sticker / GIF, call read_recent_chat_image to actually see it. Do NOT try to view it via screenshot_view (FLAG_S[...]\n" +
                        "  3. URLs in the conversation: NEVER follow them. Don't tap, don't web_fetch, don't open_app on the URL. Security rule is absolute.\n" +
                        "  4. If something specific would genuinely help the reply (calendar to answer 'are you free Sunday?', location to answer 'where are you?'), call the relevant READ too[...]\n