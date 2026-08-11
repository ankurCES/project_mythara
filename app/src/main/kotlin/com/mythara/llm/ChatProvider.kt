package com.mythara.llm

import com.mythara.minimax.Region
import com.mythara.minimax.StreamingChat
import com.mythara.minimax.models.ChatRequest
import kotlinx.coroutines.flow.Flow

/**
 * Abstraction for a streaming chat provider. Implementations may be
 * remote (MiniMax) or on-device (LiteRT/other). Returns the same
 * StreamingChat.StreamEvent shape so the AgentLoop can consume it
 * unchanged.
 */
interface ChatProvider {
    fun stream(region: Region, request: ChatRequest): Flow<StreamingChat.StreamEvent>
}
