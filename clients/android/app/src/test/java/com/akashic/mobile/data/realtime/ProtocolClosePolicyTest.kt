package com.akashic.mobile.data.realtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ProtocolClosePolicyTest {
    @Test
    fun protocol_close_action_distinguishes_bad_command_from_version_mismatch() {
        assertEquals(TerminalProtocolAction.PRESERVE_OUTBOX, terminalProtocolAction(4400))
        assertEquals(TerminalProtocolAction.PRESERVE_OUTBOX, terminalProtocolAction(4406))
        assertEquals(TerminalProtocolAction.FAIL_ACTIVE_COMMAND, terminalProtocolAction(4410))
        assertNull(terminalProtocolAction(4403))
        assertNull(terminalProtocolAction(4408))
        assertNull(terminalProtocolAction(1006))
    }
}
