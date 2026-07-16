package com.akashic.mobile

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.app.RemoteInput
import androidx.core.content.ContextCompat
import com.akashic.mobile.data.realtime.FinalMessageEvent
import com.akashic.mobile.domain.model.ConnectionPhase
import com.akashic.mobile.domain.model.ConnectionState
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import java.util.Base64

internal fun notificationIntentData(sessionId: String?, messageId: String?): String {
    fun encode(value: String?): String = value?.let {
        "1" + Base64.getUrlEncoder().withoutPadding().encodeToString(it.toByteArray(Charsets.UTF_8))
    } ?: "0"

    return "akashic://notification/open/${encode(sessionId)}/${encode(messageId)}"
}

class MobileConnectionService : Service() {
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private lateinit var app: App
    private lateinit var notificationManager: NotificationManagerCompat
    private var stateJob: Job? = null
    private var messageJob: Job? = null

    override fun onCreate() {
        super.onCreate()
        app = application as App
        notificationManager = NotificationManagerCompat.from(this)
        createNotificationChannels()
        startForeground(CONNECTION_NOTIFICATION_ID, connectionNotification(app.container.realtimeSession.state.value.connection))
        app.container.realtimeSession.start()
        observeConnection()
        observeFinalMessages()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_DISCONNECT) {
            app.container.realtimeSession.restartPairing()
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf(startId)
            return START_NOT_STICKY
        }
        if (intent?.action == ACTION_REPLY) {
            val sessionId = requireNotNull(intent.getStringExtra(EXTRA_SESSION_ID)) {
                "通知快捷回复缺少会话"
            }
            val reply = RemoteInput.getResultsFromIntent(intent)
                ?.getCharSequence(REMOTE_INPUT_REPLY_KEY)
                ?.toString()
                ?.trim()
                .orEmpty()
            if (reply.isNotEmpty()) app.container.realtimeSession.sendNotificationReply(sessionId, reply)
            return START_STICKY
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        stateJob?.cancel()
        messageJob?.cancel()
        serviceScope.cancel()
        super.onDestroy()
    }

    private fun observeConnection() {
        stateJob = serviceScope.launch {
            app.container.realtimeSession.state.collectLatest { state ->
                if (
                    Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU ||
                    ContextCompat.checkSelfPermission(
                        this@MobileConnectionService,
                        Manifest.permission.POST_NOTIFICATIONS,
                    ) == PackageManager.PERMISSION_GRANTED
                ) {
                    notificationManager.notify(
                        CONNECTION_NOTIFICATION_ID,
                        connectionNotification(state.connection),
                    )
                }
            }
        }
    }

    private fun observeFinalMessages() {
        messageJob = serviceScope.launch {
            app.container.realtimeSession.finalMessages.collect { event ->
                val currentSessionId = app.container.realtimeSession.state.value.currentSessionId
                if (
                    (
                        Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU ||
                            ContextCompat.checkSelfPermission(
                                this@MobileConnectionService,
                                Manifest.permission.POST_NOTIFICATIONS,
                            ) == PackageManager.PERMISSION_GRANTED
                    ) &&
                    notificationManager.areNotificationsEnabled() &&
                    MessageNotificationPolicy.shouldNotify(
                        appVisible = app.visibility.isVisible,
                        currentSessionId = currentSessionId,
                        event = event,
                    )
                ) {
                    notificationManager.notify(messageNotificationId(event), messageNotification(event))
                }
            }
        }
    }

    private fun createNotificationChannels() {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(
                CONNECTION_CHANNEL_ID,
                getString(R.string.notification_channel_connection),
                NotificationManager.IMPORTANCE_LOW,
            ).apply {
                description = getString(R.string.notification_channel_connection_description)
                setShowBadge(false)
                lockscreenVisibility = Notification.VISIBILITY_PRIVATE
            },
        )
        manager.createNotificationChannel(
            NotificationChannel(
                MESSAGE_CHANNEL_ID,
                getString(R.string.notification_channel_messages),
                NotificationManager.IMPORTANCE_DEFAULT,
            ).apply {
                description = getString(R.string.notification_channel_messages_description)
                lockscreenVisibility = Notification.VISIBILITY_PRIVATE
            },
        )
    }

    private fun connectionNotification(connection: ConnectionState): Notification {
        val status = when (connection.phase) {
            ConnectionPhase.READY -> "连接正常"
            ConnectionPhase.DEGRADED -> "网络不稳，正在重连"
            ConnectionPhase.CLOSED -> "连接已断开"
            ConnectionPhase.FAILED -> "连接启动失败"
            else -> "正在连接"
        }
        return NotificationCompat.Builder(this, CONNECTION_CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(getString(R.string.notification_connection_title))
            .setContentText(status)
            .setContentIntent(openAppIntent(sessionId = null))
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .setVisibility(NotificationCompat.VISIBILITY_PRIVATE)
            .setPublicVersion(privatePublicNotification())
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setSilent(true)
            .build()
    }

    private fun messageNotification(event: FinalMessageEvent): Notification =
        NotificationCompat.Builder(this, MESSAGE_CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(getString(R.string.notification_message_title))
            .setContentText(MessageNotificationPolicy.preview(event))
            .setContentIntent(openAppIntent(event.sessionId, event.messageId))
            .addAction(
                NotificationCompat.Action.Builder(
                    0,
                    getString(R.string.notification_reply_action),
                    replyIntent(event.sessionId),
                ).addRemoteInput(
                    RemoteInput.Builder(REMOTE_INPUT_REPLY_KEY)
                        .setLabel(getString(R.string.notification_reply_hint))
                        .build(),
                ).build(),
            )
            .setCategory(NotificationCompat.CATEGORY_MESSAGE)
            .setVisibility(NotificationCompat.VISIBILITY_PRIVATE)
            .setPublicVersion(privatePublicNotification())
            .setAutoCancel(true)
            .build()

    private fun privatePublicNotification(): Notification =
        NotificationCompat.Builder(this, MESSAGE_CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(getString(R.string.notification_private_title))
            .setContentText(getString(R.string.notification_private_text))
            .build()

    private fun openAppIntent(sessionId: String?, messageId: String? = null): PendingIntent {
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP
            data = android.net.Uri.parse(notificationIntentData(sessionId, messageId))
            if (sessionId != null) putExtra(EXTRA_SESSION_ID, sessionId)
            if (messageId != null) putExtra(EXTRA_MESSAGE_ID, messageId)
        }
        return PendingIntent.getActivity(
            this,
            sessionId?.hashCode() ?: CONNECTION_NOTIFICATION_ID,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
    }

    private fun replyIntent(sessionId: String): PendingIntent = PendingIntent.getService(
        this,
        sessionId.hashCode(),
        Intent(this, MobileConnectionService::class.java).apply {
            action = ACTION_REPLY
            putExtra(EXTRA_SESSION_ID, sessionId)
        },
        PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_MUTABLE,
    )

    private fun messageNotificationId(event: FinalMessageEvent): Int =
        event.messageId.hashCode().let { hash ->
            if (hash == CONNECTION_NOTIFICATION_ID) hash + 1 else hash
        }

    companion object {
        const val EXTRA_SESSION_ID = "com.akashic.mobile.extra.SESSION_ID"
        const val EXTRA_MESSAGE_ID = "com.akashic.mobile.extra.MESSAGE_ID"
        private const val ACTION_START = "com.akashic.mobile.action.START_CONNECTION"
        private const val ACTION_DISCONNECT = "com.akashic.mobile.action.DISCONNECT"
        private const val ACTION_REPLY = "com.akashic.mobile.action.REPLY"
        private const val REMOTE_INPUT_REPLY_KEY = "reply_text"
        private const val CONNECTION_CHANNEL_ID = "mobile_connection"
        private const val MESSAGE_CHANNEL_ID = "mobile_messages"
        private const val CONNECTION_NOTIFICATION_ID = 1_001

        fun start(context: Context) {
            ContextCompat.startForegroundService(
                context,
                Intent(context, MobileConnectionService::class.java).setAction(ACTION_START),
            )
        }

        fun disconnect(context: Context) {
            ContextCompat.startForegroundService(
                context,
                Intent(context, MobileConnectionService::class.java).setAction(ACTION_DISCONNECT),
            )
        }
    }
}
