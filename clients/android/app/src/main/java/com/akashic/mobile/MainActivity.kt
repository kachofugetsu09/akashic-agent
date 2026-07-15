package com.akashic.mobile

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Snackbar
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import androidx.core.content.edit
import androidx.core.net.toUri
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.akashic.mobile.ui.conversation.MobileConversationScaffold
import com.akashic.mobile.ui.design.AkashicTheme
import com.akashic.mobile.ui.pairing.PairingScreen

class MainActivity : ComponentActivity() {
    private val viewModel by viewModels<MainViewModel>()
    private val requestedSessionId = mutableStateOf<String?>(null)
    private val notificationsEnabled = mutableStateOf(true)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestedSessionId.value = intent.getStringExtra(MobileConnectionService.EXTRA_SESSION_ID)
        notificationsEnabled.value = NotificationManagerCompat.from(this).areNotificationsEnabled()
        MobileConnectionService.start(this)
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.auto(Color.TRANSPARENT, Color.TRANSPARENT),
            navigationBarStyle = SystemBarStyle.auto(Color.TRANSPARENT, Color.TRANSPARENT),
        )
        setContent {
            AkashicTheme {
                val notificationPermission = rememberLauncherForActivityResult(
                    ActivityResultContracts.RequestPermission(),
                ) { granted -> notificationsEnabled.value = granted }
                val attachmentPicker = rememberLauncherForActivityResult(
                    ActivityResultContracts.OpenMultipleDocuments(),
                ) { uris ->
                    if (uris.isNotEmpty()) viewModel.addAttachments(uris)
                }
                val session by viewModel.sessionState.collectAsStateWithLifecycle()
                val conversation by viewModel.conversationState.collectAsStateWithLifecycle()
                LaunchedEffect(Unit) {
                    if (shouldRequestNotificationPermission()) {
                        markNotificationPermissionRequested()
                        notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
                    }
                }
                LaunchedEffect(session.hasProfile) {
                    if (session.hasProfile) MobileConnectionService.start(this@MainActivity)
                }
                LaunchedEffect(session.initialized, session.hasProfile, requestedSessionId.value) {
                    val sessionId = requestedSessionId.value
                    if (session.initialized && session.hasProfile && sessionId != null) {
                        viewModel.selectSession(sessionId)
                        requestedSessionId.value = null
                    }
                }
                Box(Modifier.fillMaxSize()) {
                    if (!session.initialized) {
                        CircularProgressIndicator(Modifier.align(Alignment.Center))
                    } else if (session.hasProfile) {
                        MobileConversationScaffold(
                            state = conversation,
                            onSelectSession = viewModel::selectSession,
                            onNewSession = viewModel::createSession,
                            onRestartPairing = viewModel::restartPairing,
                            onReloadFromServer = viewModel::reloadFromServer,
                            onAttach = { attachmentPicker.launch(arrayOf("*/*")) },
                            onRemoveAttachment = viewModel::removeAttachment,
                            onRetryAttachment = viewModel::retryAttachment,
                            onRetryDownloadedAttachment = viewModel::retryDownloadedAttachment,
                            onOpenDownloadedAttachment = viewModel::touchDownloadedAttachment,
                            onDismissError = viewModel::dismissError,
                            onSend = viewModel::sendMessage,
                            onStop = viewModel::stopCurrentTurn,
                        )
                    } else {
                        PairingScreen(
                            scanGeneration = session.scanGeneration,
                            confirmationCode = session.pairingConfirmationCode,
                            errorMessage = session.errorMessage,
                            onQrCode = viewModel::onQrCode,
                        )
                    }
                    if (!notificationsEnabled.value && notificationPermissionWasRequested()) {
                        NotificationPermissionNotice(
                            modifier = Modifier
                                .align(Alignment.BottomCenter)
                                .padding(horizontal = 16.dp, vertical = 20.dp),
                            onOpenSettings = ::openNotificationSettings,
                        )
                    }
                }
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        requestedSessionId.value = intent.getStringExtra(MobileConnectionService.EXTRA_SESSION_ID)
    }

    override fun onResume() {
        super.onResume()
        notificationsEnabled.value = NotificationManagerCompat.from(this).areNotificationsEnabled()
    }

    private fun shouldRequestNotificationPermission(): Boolean =
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) !=
            PackageManager.PERMISSION_GRANTED &&
            !notificationPermissionWasRequested()

    private fun notificationPermissionWasRequested(): Boolean =
        getSharedPreferences(PERMISSION_PREFERENCES, MODE_PRIVATE)
            .getBoolean(KEY_NOTIFICATION_PERMISSION_REQUESTED, false)

    private fun markNotificationPermissionRequested() {
        getSharedPreferences(PERMISSION_PREFERENCES, MODE_PRIVATE)
            .edit { putBoolean(KEY_NOTIFICATION_PERMISSION_REQUESTED, true) }
    }

    private fun openNotificationSettings() {
        startActivity(
            Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS).apply {
                data = "package:$packageName".toUri()
                putExtra(Settings.EXTRA_APP_PACKAGE, packageName)
            },
        )
    }

    private companion object {
        const val PERMISSION_PREFERENCES = "notification_permission"
        const val KEY_NOTIFICATION_PERMISSION_REQUESTED = "requested"
    }
}

@Composable
private fun NotificationPermissionNotice(
    modifier: Modifier = Modifier,
    onOpenSettings: () -> Unit,
) {
    Snackbar(
        modifier = modifier,
        action = {
            TextButton(onClick = onOpenSettings) {
                Text(stringResource(R.string.notification_permission_action))
            }
        },
    ) {
        Text(stringResource(R.string.notification_permission_notice))
    }
}
