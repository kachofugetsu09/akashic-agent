package com.akashic.mobile.data.realtime

import com.akashic.mobile.domain.model.ConnectionPhase
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ConnectionRecoveryPolicyTest {
    private val unavailable = TransferNetworkState(TransferNetworkKind.UNAVAILABLE, false)
    private val unmetered = TransferNetworkState(TransferNetworkKind.UNMETERED, true)

    @Test
    fun `failure before recovery consumes one immediate reconnect`() {
        val latch = NetworkRecoveryLatch()
        latch.onGenerationStarted(7, unmetered)
        latch.onNetworkState(7, unmetered, unavailable, hasConnectionTarget = true)

        assertFalse(latch.consume(7))
        latch.onNetworkState(7, unavailable, unmetered, hasConnectionTarget = true)
        assertTrue(latch.consume(7))
        assertFalse(latch.consume(7))
    }

    @Test
    fun `recovery before failure survives duplicate callbacks and consumes once`() {
        val latch = NetworkRecoveryLatch()
        latch.onGenerationStarted(11, unmetered)
        latch.onNetworkState(11, unmetered, unavailable, hasConnectionTarget = true)
        latch.onNetworkState(11, unavailable, unmetered, hasConnectionTarget = true)
        latch.onNetworkState(11, unmetered, unmetered, hasConnectionTarget = true)

        assertTrue(latch.consume(11))
        latch.onNetworkState(11, unmetered, unmetered, hasConnectionTarget = true)
        assertFalse(latch.consume(11))
    }

    @Test
    fun `recovery owner does not cross generation or healthy progress`() {
        val latch = NetworkRecoveryLatch()
        latch.onGenerationStarted(17, unmetered)
        latch.onNetworkState(17, unmetered, unavailable, hasConnectionTarget = true)
        latch.onNetworkState(17, unavailable, unmetered, hasConnectionTarget = true)

        assertFalse(latch.consume(18))
        latch.onConnectionProgress(17)
        assertFalse(latch.consume(17))

        latch.onNetworkState(18, unmetered, unavailable, hasConnectionTarget = false)
        latch.onNetworkState(18, unavailable, unmetered, hasConnectionTarget = false)
        assertFalse(latch.consume(18))
    }

    @Test
    fun `phase deadlines distinguish handshake authentication and sync progress`() {
        assertEquals(10_000L, ConnectionDeadlinePhase.CHALLENGE.deadlineMillis())
        assertEquals(10_000L, ConnectionDeadlinePhase.AUTHENTICATION.deadlineMillis())
        assertEquals(20_000L, ConnectionDeadlinePhase.SYNC.deadlineMillis())
        assertTrue(ConnectionDeadlinePhase.CHALLENGE.timeoutMessage().contains("握手"))
        assertTrue(ConnectionDeadlinePhase.AUTHENTICATION.timeoutMessage().contains("认证"))
        assertTrue(ConnectionDeadlinePhase.SYNC.timeoutMessage().contains("同步"))
    }

    @Test
    fun `continuous sync replay keeps idle deadline beyond twenty seconds`() {
        var expiresAt = ConnectionDeadlinePhase.SYNC.deadlineMillis()
        for (progressAt in listOf(19_000L, 38_000L, 57_000L)) {
            assertTrue(progressAt < expiresAt)
            assertTrue(
                shouldRefreshSyncDeadline(
                    phaseBeforeFrame = ConnectionPhase.SYNCING,
                    phaseAfterFrame = ConnectionPhase.SYNCING,
                )
            )
            expiresAt = progressAt + ConnectionDeadlinePhase.SYNC.deadlineMillis()
        }
        assertTrue(expiresAt > 60_000L)
        assertFalse(
            shouldRefreshSyncDeadline(
                phaseBeforeFrame = ConnectionPhase.SYNCING,
                phaseAfterFrame = ConnectionPhase.DEGRADED,
            )
        )
    }

    @Test
    fun `late candidates cannot downgrade connection progress`() {
        assertFalse(
            shouldApplyCandidateOpen(
                connectionPhase = ConnectionPhase.DEVICE_PROOF,
                hasActiveCandidate = false,
                candidateGeneration = 7,
                pairingConfirmationGeneration = null,
            )
        )
        assertFalse(
            shouldApplyCandidateOpen(
                connectionPhase = ConnectionPhase.SYNCING,
                hasActiveCandidate = true,
                candidateGeneration = 7,
                pairingConfirmationGeneration = null,
            )
        )
        assertFalse(
            shouldApplyCandidateOpen(
                connectionPhase = ConnectionPhase.SERVER_CHALLENGE,
                hasActiveCandidate = false,
                candidateGeneration = 7,
                pairingConfirmationGeneration = 7,
            )
        )
        assertTrue(
            shouldApplyCandidateOpen(
                connectionPhase = ConnectionPhase.DEGRADED,
                hasActiveCandidate = false,
                candidateGeneration = 8,
                pairingConfirmationGeneration = 7,
            )
        )

        val authentication = ConnectionPhaseDeadline(
            generation = 7,
            phase = ConnectionDeadlinePhase.AUTHENTICATION,
        )
        assertFalse(
            shouldReplacePhaseDeadline(
                current = authentication,
                generation = 7,
                next = ConnectionDeadlinePhase.CHALLENGE,
            )
        )
        assertTrue(
            shouldReplacePhaseDeadline(
                current = authentication,
                generation = 7,
                next = ConnectionDeadlinePhase.SYNC,
            )
        )
    }

    @Test
    fun `rejected loser open cannot consume recovered outage`() {
        val latch = NetworkRecoveryLatch()
        latch.onGenerationStarted(23, unmetered)
        latch.onNetworkState(23, unmetered, unavailable, hasConnectionTarget = true)
        latch.onNetworkState(23, unavailable, unmetered, hasConnectionTarget = true)

        val accepted = shouldApplyCandidateOpen(
            connectionPhase = ConnectionPhase.SYNCING,
            hasActiveCandidate = true,
            candidateGeneration = 23,
            pairingConfirmationGeneration = null,
        )
        if (accepted) latch.onConnectionProgress(23)

        assertFalse(accepted)
        assertTrue(latch.consume(23))
        assertFalse(latch.consume(23))
    }
}
