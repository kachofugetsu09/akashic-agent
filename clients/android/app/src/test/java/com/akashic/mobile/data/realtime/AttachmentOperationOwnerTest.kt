package com.akashic.mobile.data.realtime

import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AttachmentOperationOwnerTest {
    @Test
    fun `send result reports success exactly once`() = runBlocking {
        val results = mutableListOf<Boolean>()

        withSendResult(results::add) { report -> report(true) }

        assertEquals(listOf(true), results)
    }

    @Test
    fun `send result reports failure before propagating internal error`() = runBlocking {
        val results = mutableListOf<Boolean>()
        val failure = runCatching {
            withSendResult(results::add) { error("persistence failed") }
        }.exceptionOrNull()

        assertEquals(listOf(false), results)
        assertEquals("persistence failed", failure?.message)
    }

    @Test
    fun `draft identity rejects removal instead of degrading to text`() {
        assertTrue(attachmentDraftMatchesExpected(listOf("a", "b"), listOf("b", "a")))
        assertFalse(attachmentDraftMatchesExpected(emptyList(), listOf("a")))
        assertFalse(attachmentDraftMatchesExpected(listOf("a"), listOf("a", "a")))
    }
}
