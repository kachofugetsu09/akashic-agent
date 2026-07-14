package com.akashic.mobile.data.local

import androidx.room.withTransaction

class LocalDeliveryStore(private val database: AppDatabase) {
    suspend fun advanceCursor(
        deviceId: String,
        throughEventSeq: Long,
        connectionEpoch: Long,
        updatedAt: Long,
    ) {
        val changed = database.withTransaction {
            database.realtimeCursors().advance(
                deviceId = deviceId,
                throughEventSeq = throughEventSeq,
                connectionEpoch = connectionEpoch,
                updatedAt = updatedAt,
            )
        }
        check(changed == 1) { "Cursor rollback or stale connection epoch for device $deviceId" }
    }

    suspend fun markOutboxAttempt(commandId: String, attemptedAt: Long) {
        val changed = database.outbox().markInFlight(commandId, attemptedAt)
        check(changed == 1) { "Outbox command is not pending: $commandId" }
    }
}
