package com.mythara.data

import android.content.Context
import android.util.Base64
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.google.crypto.tink.Aead
import com.google.crypto.tink.KeyTemplates
import com.google.crypto.tink.aead.AeadConfig
import com.google.crypto.tink.integration.android.AndroidKeysetManager
import com.mythara.minimax.Region
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Encrypted persistence for the user's MiniMax API key + chosen region +
 * model. Uses Jetpack DataStore (preferences flavour) plus Google Tink
 * AEAD with the wrapping key in the Android Keystore — the modern
 * replacement for the now-deprecated EncryptedSharedPreferences.
 *
 * - The API key is base64(AEAD-encrypted-with-Tink) stored under
 *   `apiKey.encrypted`. Tink's [AndroidKeysetManager] handles key
 *   generation, rotation, and Keystore-backed wrapping.
 * - Region and model are stored in plaintext — not sensitive, and
 *   needed pre-decryption to render Settings.
 * - The Tink keyset itself lives in `mythara_master_keyset` SharedPreferences
 *   wrapped by an Android Keystore key with alias `mythara_master_key`.
 */
@Singleton
class SettingsStore @Inject constructor(
    @ApplicationContext private val ctx: Context,
) {
    private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "settings")

    private val keyApiKeyEncrypted = stringPreferencesKey("apiKey.encrypted")
    /** Captured MiniMax web-session cookies (encrypted JSON
             *  `{token, groupId, expiresAtMs}`). Used by MiniMaxUsageClient
     *  to authenticate against the same surface the platform web
     *  dashboard uses, since Bearer auth gives a different scoped
     *  view than the user's signed-in browser session. */
    private val keyMiniMaxWebSessionEncrypted = stringPreferencesKey("miniMaxWebSession.encrypted")
    private val keyRegion          = stringPreferencesKey("region")
    private val keyModel           = stringPreferencesKey("model")
    // Gemini Developer API key — optional secondary credential used
    // exclusively for the take_photo vision route. When set, VisionService
    // routes captured images through Gemini instead of MiniMax-VL-01.
    // Lives in the same Tink-encrypted DataStore as the MiniMax key so
    // it inherits the same Keystore-backed at-rest protection.
    private val keyGeminiKeyEncrypted = stringPreferencesKey("geminiKey.encrypted")
    // ElevenLabs TTS: optional API key + toggle to route Tts.speak()
    // through their hosted voice synthesis instead of Android's
    // built-in TextToSpeech. Key is Tink-AEAD-encrypted like the
    // other API keys; voice id + toggle are plaintext prefs.
    private val keyElevenLabsKeyEncrypted = stringPreferencesKey("elevenLabsKey.encrypted")
    private val keyElevenLabsVoiceId = stringPreferencesKey("elevenLabsVoiceId")
    private val keyUseElevenLabs = booleanPreferencesKey("useElevenLabs")
    // Vision routing — when true, VisionService tries the cloud
    // backends (Gemini → MiniMax-VL) before the on-device Gemma
    // path. Default is false (Gemma first) for privacy + zero-cost
    // captioning; users with a Gemini key who prefer higher
    // accuracy can flip this from Settings.
    private val keyPreferCloudVision = booleanPreferencesKey("preferCloudVision")
    // Supertonic-2 voice name (F1..F5 / M1..M5). The engine loads
    // the matching <name>.json voice style on every speak() so
    // changes apply immediately without restarting the app.
    private val keySupertonicVoice = stringPreferencesKey("supertonicVoice")

    private val aead: Aead by lazy {
        AeadConfig.register()
        AndroidKeysetManager.Builder()
            .withSharedPref(ctx, "mythara_master_keyset", "mythara_master_key")
            .withKeyTemplate(KeyTemplates.get("AES256_GCM"))
            .withMasterKeyUri("android-keystore://mythara_master_key")
            .build()
            .keysetHandle
            .getPrimitive(Aead::class.java)
    }
      fun apiKeyFlow(): Flow<String?> = ctx.dataStore.data.map { prefs ->
        prefs[keyApiKeyEncrypted]?.let { tryDecrypt(it) }
    }

    fun regionFlow(): Flow<Region> = ctx.dataStore.data.map { prefs ->
        Region.fromId(prefs[keyRegion])
    }

    fun modelFlow(): Flow<String> = ctx.dataStore.data.map { prefs ->
        prefs[keyModel] ?: DEFAULT_MODEL
    }

    suspend fun setApiKey(plain: String) {
        val ct = aead.encrypt(plain.toByteArray(Charsets.UTF_8), null)
        ctx.dataStore.edit { it[keyApiKeyEncrypted] = Base64.encodeToString(ct, Base64.NO_WRAP) }
    }

    /** Stored MiniMax web-session, decrypted. Returns null when
     *  the user hasn't completed the WebView sign-in or the stored
     *  session has expired (caller checks expiresAtMs). */
    @kotlinx.serialization.Serializable
    data class MiniMaxWebSession(
        val token: String,
        val groupId: String,
        val expiresAtMs: Long,
    )

    private val webSessionJson = kotlinx.serialization.json.Json {
        ignoreUnknownKeys = true; explicitNulls = false
    }

    suspend fun miniMaxWebSession(): MiniMaxWebSession? {
        val raw = ctx.dataStore.data.first()[keyMiniMaxWebSessionEncrypted] ?: return null
        val plain = tryDecrypt(raw) ?: return null
        return runCatching {
            webSessionJson.decodeFromString(MiniMaxWebSession.serializer(), plain)
        }.getOrNull()
    }

    suspend fun setMiniMaxWebSession(session: MiniMaxWebSession) {
        val plain = webSessionJson.encodeToString(MiniMaxWebSession.serializer(), session)
        val ct = aead.encrypt(plain.toByteArray(Charsets.UTF_8), null)
             ctx.dataStore.edit { it[keyMiniMaxWebSessionEncrypted] = Base64.encodeToString(ct, Base64.NO_WRAP) }
    }

    suspend fun clearMiniMaxWebSession() {
        ctx.dataStore.edit { it.remove(keyMiniMaxWebSessionEncrypted) }
    }

    /**
     * Read the Gemini vision key. Same Tink AEAD path as the MiniMax key.
     * Returns null if not set.
     */
    fun geminiKeyFlow(): Flow<String?> = ctx.dataStore.data.map { prefs ->
        prefs[keyGeminiKeyEncrypted]?.let { tryDecrypt(it) }
    }

    suspend fun setGeminiKey(plain: String) {
        if (plain.isBlank()) {
            ctx.dataStore.edit { it.remove(keyGeminiKeyEncrypted) }
            return
        }
        val ct = aead.encrypt(plain.toByteArray(Charsets.UTF_8), null)
        ctx.dataStore.edit { it[keyGeminiKeyEncrypted] = Base64.encodeToString(ct, Base64.NO_WRAP) }
    }

    suspend fun clearGeminiKey() {
        ctx.dataStore.edit { it.remove(keyGeminiKeyEncrypted) }
    }

    // ---------- ElevenLabs ----------

    fun elevenLabsKeyFlow(): Flow<String?> = ctx.dataStore.data.map { prefs ->
        prefs[keyElevenLabsKeyEncrypted]?.let { tryDecrypt(it) }
    }

    fun elevenLabsVoiceIdFlow(): Flow<String> = ctx.dataStore.data.map { prefs ->
        prefs[keyElevenLabsVoiceId] ?: DEFAULT_ELEVEN_LABS_VOICE_ID
    }

    fun useElevenLabsFlow(): Flow<Boolean> = ctx.dataStore.data.map { prefs ->
        prefs[keyUseElevenLabs] ?: false
    }

    suspend fun setElevenLabsKey(plain: String) {
              if (plain.isBlank()) {
            ctx.dataStore.edit { it.remove(keyElevenLabsKeyEncrypted) }
            return
        }
        val ct = aead.encrypt(plain.toByteArray(Charsets.UTF_8), null)
        ctx.dataStore.edit { it[keyElevenLabsKeyEncrypted] = Base64.encodeToString(ct, Base64.NO_WRAP) }
    }

    suspend fun clearElevenLabsKey() {
        ctx.dataStore.edit { it.remove(keyElevenLabsKeyEncrypted) }
    }

    suspend fun setElevenLabsVoiceId(voiceId: String) {
        val v = voiceId.trim()
        ctx.dataStore.edit {
            if (v.isBlank()) it.remove(keyElevenLabsVoiceId) else it[keyElevenLabsVoiceId] = v
        }
    }

    suspend fun setUseElevenLabs(value: Boolean) {
        ctx.dataStore.edit { it[keyUseElevenLabs] = value }
    }

    suspend fun setPreferCloudVision(value: Boolean) {
        ctx.dataStore.edit { it[keyPreferCloudVision] = value }
    }

    suspend fun setSupertonicVoice(name: String) {
        val v = name.trim()
        ctx.dataStore.edit {
            if (v.isBlank()) it.remove(keySupertonicVoice) else it[keySupertonicVoice] = v
        }
    }

    suspend fun setRegion(region: Region) {
        ctx.dataStore.edit { it[keyRegion] = region.name }
    }

    suspend fun setModel(model: String) {
        ctx.dataStore.edit { it[keyModel] = model }
    }

    /** Convenience: a snapshot of the trio for the network layer. */
    suspend fun snapshot(): Snapshot {
        val prefs = ctx.dataStore.data.first()
        return Snapshot(
            apiKey = prefs[keyApiKeyEncrypted]?.let { tryDecrypt(it) },
            region = Region.fromId(prefs[keyRegion]),
            model = prefs[keyModel] ?: DEFAULT_MODEL,
            geminiKey = prefs[keyGeminiKeyEncrypted]?.let { tryDecrypt(it) },
            elevenLabsKey = prefs[keyElevenLabsKeyEncrypted]?.let { tryDecrypt(it) },
            elevenLabsVoiceId = prefs[keyElevenLabsVoiceId] ?: DEFAULT_ELEVEN_LABS_VOICE_ID,
            useElevenLabs = prefs[keyUseElevenLabs] ?: false,
            preferCloudVision = prefs[keyPreferCloudVision] ?: false,
            supertonicVoice = prefs[keySupertonicVoice] ?: DEFAULT_SUPERTONIC_VOICE,
        )
    }

    private fun tryDecrypt(b64: String): String? = runCatching {
        val pt = aead.decrypt(Base64.decode(b64, Base64.NO_WRAP), null)
        String(pt, Charsets.UTF_8)
    }.getOrNull()

    data class Snapshot(
        val apiKey: String?,
        val region: Region,
        val model: String,
        /** Optional Gemini vision key. Null means we fall back to MiniMax-VL. */
        val geminiKey: String? = null,
        /** Optional ElevenLabs TTS key. Null disables the ElevenLabs route. */
        val elevenLabsKey: String? = null,
        /** ElevenLabs voice id; defaults to a stock voice if unset. */
        val elevenLabsVoiceId: String = DEFAULT_ELEVEN_LABS_VOICE_ID,
        /** When true AND key is set, Tts.speak routes through ElevenLabs. */
        val useElevenLabs: Boolean = false,
        /** Vision routing: false (default) = on-device Gemma 4 E2B
         *  first → cloud Gemini → MiniMax-VL. True flips the order
         *  so cloud Gemini runs first (when the key is configured),
         *  with Gemma + MiniMax as fallbacks. */
        val preferCloudVision: Boolean = false,
        /** Supertonic-2 voice id (F1..F5, M1..M5). Defaults to M1. */
        val supertonicVoice: String = DEFAULT_SUPERTONIC_VOICE,
    )

    companion object {
        /**
               * Default = M2.7. The function-calling guide
         * (platform.minimax.io/docs/guides/text-m2-function-call) explicit
         * cites M2.7 for "exceptional Tool Use capabilities" — the right
         * default for an agentic runtime. Users who want faster/cheaper
         * can pick a highspeed or older variant in Settings.
         */
        const val DEFAULT_MODEL: String = "MiniMax-M2.7"

        /**
         * Models documented on /v1/chat/completions per the OpenAI-compat
         * spec page (text-chat-openai.md). M1 and VL-01 aren't listed on
         * this endpoint and are intentionally excluded.
         */
        val SUPPORTED_MODELS: List<String> = listOf(
            // GROQ (free tier — 30 RPM / 6K TPM / 1K RPD)
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.6-27b",
            "groq/compound",
            "groq/compound-mini",

            // OPENROUTER :free (15 model — valid Juli 2026)
            "openrouter/free",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
            "nvidia/nemotron-3-nano-30b-a3b:free",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            "nvidia/nemotron-nano-12b-v2-vl:free",
            "nvidia/nemotron-nano-9b-v2:free",
            "nvidia/nemotron-3.5-content-safety:free",
            "google/gemma-4-31b-it:free",
            "google/gemma-4-26b-a4b-it:free",
            "openai/gpt-oss-20b:free",
            "cohere/north-mini-code:free",
            "poolside/laguna-m.1:free",
            "poolside/laguna-xs-2.1:free",
            "tencent/hy3:free",

            // SAMBANOVA (free tier — 10-30 RPM, context 8K-64K)
            "Meta-Llama-3.3-70B-Instruct",
            "DeepSeek-V3.2",
            "DeepSeek-V3.1",
            "gemma-4-31B-it",
            "gpt-oss-120b",
            "MiniMax-M2.7",

            // Z.AI / GLM (free tier — ~1.000 req/hari, context 200K)
            "glm-4.7",
            "glm-4.6",
            "glm-4.5",
            "glm-4.5-air",
            "glm-5",
            "glm-5-turbo",
            "glm-5.1",
            "glm-5.2",

            // CEREBRAS (free tier — 1M token/hari, context 8K/131K)
            "gpt-oss-120b",
            "glm-4.7",

            // GOOGLE AI STUDIO (free tier — ~1.500 req/hari, 1M token/min)
            "gemini-2.0-flash-exp",
            "gemini-2.0-flash-lite-preview-02-05",
            "gemini-1.5-flash",
        )

        /**
         * Default ElevenLabs voice id — "Rachel", their long-standing
         * stock voice that's available on the free tier. Users can
         * pick a different voice id (any from their /v1/voices list)
         * via Settings.
         */
        const val DEFAULT_ELEVEN_LABS_VOICE_ID: String = "21m00Tcm4TlvDq8ikWAM"

        /** Default on-device voice id when the user hasn't picked
         *  one. M1 — male, neutral; matches what the engine
         *  shipped with before the picker. */
        const val DEFAULT_SUPERTONIC_VOICE: String = "M1"
    }
}
