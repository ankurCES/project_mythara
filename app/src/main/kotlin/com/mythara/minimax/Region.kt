package com.mythara.minimax

enum class Region(val label: String, val baseUrl: String) {
    Global("Global (minimax.io)", "https://api.minimax.io/v1/"),
    China ("China (minimaxi.com)", "https://api.minimaxi.com/v1/"),
    Groq("Groq (free tier)", "https://api.groq.com/openai/v1/"),
    OpenRouter("OpenRouter (free tier)", "https://openrouter.ai/api/v1/"),
    SambaNova("SambaNova (free tier)", "https://api.sambanova.ai/v1/"),
    Zai("z.ai / GLM (free tier)", "https://api.z.ai/api/paas/v4"),
    Cerebras("Cerebras (free tier)", "https://api.cerebras.ai/v1/"),
    GoogleAI("Google AI Studio (free tier)", "https://generativelanguage.googleapis.com/v1beta");

    companion object {
        val Default: Region = Global
        fun fromId(id: String?): Region = entries.firstOrNull { it.name == id } ?: Default
    }
}
