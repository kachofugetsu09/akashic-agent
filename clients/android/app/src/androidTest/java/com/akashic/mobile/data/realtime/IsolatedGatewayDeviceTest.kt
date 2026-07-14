package com.akashic.mobile.data.realtime

import android.util.Base64
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.akashic.mobile.App
import com.akashic.mobile.data.local.MessageWithBlocks
import com.akashic.mobile.domain.model.ConnectionPhase
import java.io.File
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class IsolatedGatewayDeviceTest {
    @Test
    fun pairSendAndReceiveFixedMedia() = runBlocking {
        val arguments = InstrumentationRegistry.getArguments()
        val offer = String(
            Base64.decode(requireNotNull(arguments.getString("pairingOfferBase64")), Base64.DEFAULT),
            Charsets.UTF_8,
        )
        val sessionId = requireNotNull(arguments.getString("historySessionId"))
        val app = ApplicationProvider.getApplicationContext<App>()
        val session = app.container.realtimeSession

        session.start()
        withTimeout(TIMEOUT_MILLIS) { session.state.first { it.initialized } }
        session.beginPairing(offer)
        val ready = withTimeout(TIMEOUT_MILLIS) {
            session.state.first { it.hasProfile && it.connection.phase == ConnectionPhase.READY }
        }
        val serverId = requireNotNull(ready.serverId)
        withTimeout(TIMEOUT_MILLIS) {
            app.container.database.conversations().observeForServer(serverId)
                .first { conversations -> conversations.any { it.sessionId == sessionId } }
        }

        session.selectSession(sessionId)
        withTimeout(TIMEOUT_MILLIS) { session.state.first { it.currentSessionId == sessionId } }
        val initial = graph(app, sessionId) { it.size >= 2 }
        assertEquals(1, initial.count { it.message.text == "这是隔离 Gateway 的历史消息" })
        assertEquals(1, initial.count { it.message.text == "历史同步成功后应只出现一次。" })

        session.sendMessage("Android 隔离端到端")
        val received = graph(app, sessionId) { messages ->
            messages.any { message ->
                message.message.text == "隔离 Gateway 已收到消息，这是固定媒体回复。" &&
                    message.attachmentLinks.any { it.attachment.state == "cached" }
            }
        }
        val reply = received.single {
            it.message.text == "隔离 Gateway 已收到消息，这是固定媒体回复。"
        }
        val attachment = reply.attachmentLinks.single().attachment
        assertEquals("image/gif", attachment.contentType)
        assertTrue(File(requireNotNull(attachment.cachePath)).isFile)
    }

    @Test
    fun processRestartResumesWithoutHistoryDuplicates() = runBlocking {
        val sessionId = requireNotNull(
            InstrumentationRegistry.getArguments().getString("historySessionId"),
        )
        val app = ApplicationProvider.getApplicationContext<App>()
        val session = app.container.realtimeSession

        session.start()
        withTimeout(TIMEOUT_MILLIS) {
            session.state.first { it.hasProfile && it.connection.phase == ConnectionPhase.READY }
        }
        session.selectSession(sessionId)
        val restored = graph(app, sessionId) { messages ->
            messages.any { it.message.text == "隔离 Gateway 已收到消息，这是固定媒体回复。" }
        }

        assertEquals(1, restored.count { it.message.text == "这是隔离 Gateway 的历史消息" })
        assertEquals(1, restored.count { it.message.text == "历史同步成功后应只出现一次。" })
        assertEquals(
            1,
            restored.count { it.message.text == "隔离 Gateway 已收到消息，这是固定媒体回复。" },
        )
    }

    private suspend fun graph(
        app: App,
        sessionId: String,
        predicate: (List<MessageWithBlocks>) -> Boolean,
    ): List<MessageWithBlocks> = withTimeout(TIMEOUT_MILLIS) {
        app.container.database.messages().observeMessageGraph(sessionId).first(predicate)
    }

    private companion object {
        const val TIMEOUT_MILLIS = 30_000L
    }
}
