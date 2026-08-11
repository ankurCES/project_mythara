package com.mythara.llm

import com.mythara.minimax.MiniMaxClient
import com.mythara.minimax.Region
import com.mythara.minimax.StreamingChat
import com.mythara.minimax.models.ChatRequest
import kotlinx.coroutines.flow.Flow

/** Simple adapter that delegates to the existing StreamingChat. */
class MiniMaxAdapter(private val client: MiniMaxClient) : ChatProvider {
    override fun stream(region: Region, request: ChatRequest): Flow<StreamingChat.StreamEvent> {
        val streaming = StreamingChat(client)
        return streaming.stream(region, request)
    }
}
