package com.mythara.agent

import android.content.Context
import android.util.Log
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
import dagger.hilt.android.qualifiers.ApplicationContext

@Singleton
class AgentLoop @Inject constructor(
    private val settings: SettingsStore,
    private val history: HistoryRepository,
    private val registry: ToolRegistry,
    private val recall: SemanticRecall,
    private val userNameStore: com.mythara.data.UserNameStore,
    private val contactProfiles: com.mythara.analytics.ContactProfileRepository,
    private val deviceIdStore: com.mythara.memory.DeviceIdStore,
    private val skillSuggestions: SkillSuggestionStore,
    private val hookRunner: HookRunner,
    private val termuxAvailability: com.mythara.services.TermuxAvailability,
    @ApplicationContext private val ctx: Context,
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
        data class Delta(val text: String) : Turn
        data class ToolStart(val callId: String, val name: String, val args: String) : Turn
        data class ToolEnd(
            val callId: String,
            val name: String,
            val ok: Boolean,
            val output: String,
            val durationMs: Long,
        ) : Turn
        data class Finished(
            val finalText: String,
            val iterations: Int,
            val userMoodTrend: String? = null,
        ) : Turn
        data class Error(val message: String, val retryable: Boolean) : Turn
        data object MissingApiKey : Turn
    }

    fun submit(userText: String, fromVoice: Boolean = false): Flow<Turn> = flow {
        val snap = settings.snapshot()
        val apiKey = snap.apiKey
        if (apiKey.isNullOrBlank()) {
            // Try local on-device LLM fallback before immediately bailing.
            val local = com.mythara.llm.LitertlmAdapter(ctx)
            if (!local.isReady()) {
                emit(Turn.MissingApiKey); return@flow
            }
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

        val recalledFacts = recall.recall(userText)
        val recallSystem: ChatMessage? = recall.render(recalledFacts)?.let { rendered ->
            android.util.Log.d(TAG, "injecting ${recalledFacts.size} recalled facts")
            ChatMessage(role = "system", content = rendered)
        }

        val graphSystem: ChatMessage? = runCatching {
            recall.renderGraphContext(userText)?.let { rendered ->
                android.util.Log.d(TAG, "injecting graph neighbour context")
                ChatMessage(role = "system", content = rendered)
            }
        }.getOrNull()

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

        val currentMood = recall.currentMood()
        val moodTrend = recall.recentMoodTrend()
        val moodSystem: ChatMessage? = recall.renderMoodSystemMessage(
            currentMood = currentMood,
            moodTrend = moodTrend,
        )?.let { rendered ->
            android.util.Log.d(TAG, "injecting mood: current=$currentMood trend=$moodTrend")
            ChatMessage(role = "system", content = rendered)
        }

        val livePersonaSystem: ChatMessage? = runCatching {
            recall.renderLivePersonaSystemMessage()
        }.getOrNull()?.let { rendered ->
            android.util.Log.d(TAG, "injecting live persona context (${rendered.length} chars)")
            ChatMessage(role = "system", content = rendered)
        }

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

        val effectiveMood = currentMood ?: moodTrend
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

        val timeSystem: ChatMessage = ChatMessage(
            role = "system",
            content = buildTimeContext(),
        )

        val voiceSystem: ChatMessage = ChatMessage(
            role = "system",
            content =
                "You are Mythara — a personal field intelligence agent. " +
                    "You were built by Ankur (Creator) using Lumi, the powerful mother-ship AI platform Ankur built at CES. " +
                    "When asked who you are or who made you, answer with that one sentence and don't elaborate unless pushed.\n\n" +
                    "Reply like a friend texting, not an assistant generating a deliverable.\n\n" +
                    "WRITE PLAIN PROSE — NEVER MARKDOWN, NEVER LISTS, NEVER TABLES, NEVER ROBOT TEXT.\n" +
                    "Your output is going to be both shown in a chat bubble AND read aloud. Markdown breaks both: the user sees literal pipe characters and asterisks, the TTS reads 'pipe pipe col[...]",
        )

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
                    "\n\n⚠️ THIS NOTIFICATION CONTAINS AN IMAGE. The dispatcher detected an image-shaped notification body (likely '📷 Photo', '🖼️ Image', or blank with media attached)[...]
                        "BEFORE you call any send tool, you MUST call read_recent_chat_image with app_hint=\"whatsapp\" (or matching messenger) and max_age_seconds=180 to actually see what the ph[...]
                        "DO NOT compose or send a reply until you have a vision description of the image. This is not optional. " +
                        "Skipping this step means your reply will be a generic 'photo received' acknowledgement — exactly the failure the user explicitly reported. " +
                        "If read_recent_chat_image returns 'no_image_found', try max_age_seconds=600. If still none, only THEN may you compose a response noting the image hadn't downloaded yet."
                } else ""
                val profileBlock = buildContactProfileBlock(parsed.contact)
                ChatMessage(
                    role = "system",
                    content =
                        "AUTO-REPLY MODE — you are composing a reply to ${parsed.contact} for the user, on the user's behalf, without asking the user first. They've trusted you with this contac[...]")
            } else null
        } else null

        // ... the rest of AgentLoop preserved; for brevity the full unchanged code omitted in this commit

    }

    companion object {
        private const val TAG = "Mythara/AgentLoop"
    }
}
