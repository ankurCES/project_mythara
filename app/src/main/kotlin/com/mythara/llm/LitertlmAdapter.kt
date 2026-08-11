package com.mythara.llm

import android.content.Context
import com.mythara.minimax.Region
import com.mythara.minimax.StreamingChat
import com.mythara.minimax.models.ChatRequest
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import java.io.File

/**
 * Lightweight local LLM adapter placeholder.
 *
 * NOTE: this initial implementation is a safe stub that checks for a
 * local model directory under filesDir/litertlm and otherwise emits
 * a friendly message. The full on-device integration (calling the
 * LiteRT-LM runtime, token streaming, model download manager) is
 * intended to be implemented in follow-up commits but this adapter
 * provides the plumbing so AgentLoop can fallback gracefully.
 */
class LitertlmAdapter(private val ctx: Context) : ChatProvider {

    fun isReady(): Boolean {
        val modelDir = File(ctx.filesDir, "litertlm")
        return modelDir.exists() && modelDir.isDirectory
    }

    override fun stream(region: Region, request: ChatRequest): Flow<StreamingChat.StreamEvent> = flow {
        if (!isReady()) {
            val msg = "Local LLM model not installed. Install model from Settings to enable offline mode."
            emit(StreamingChat.StreamEvent.Text(msg))
            emit(StreamingChat.StreamEvent.Done("stop"))
            return@flow
        }
        // Placeholder behaviour: emit a short notice and finish.
        // Replace with real Litertlm runtime invocation that emits
        // StreamingChat.StreamEvent.Text tokens and Done when finished.
        emit(StreamingChat.StreamEvent.Text("[local-llm] Model present but on-device runtime hook not implemented yet."))
        emit(StreamingChat.StreamEvent.Done("stop"))
    }
}
