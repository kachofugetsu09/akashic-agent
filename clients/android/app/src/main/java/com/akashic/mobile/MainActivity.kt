package com.akashic.mobile

import android.graphics.Color
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.getValue
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.akashic.mobile.ui.conversation.MobileConversationScaffold
import com.akashic.mobile.ui.design.AkashicTheme
import com.akashic.mobile.ui.pairing.PairingScreen

class MainActivity : ComponentActivity() {
    private val viewModel by viewModels<MainViewModel>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.auto(Color.TRANSPARENT, Color.TRANSPARENT),
            navigationBarStyle = SystemBarStyle.auto(Color.TRANSPARENT, Color.TRANSPARENT),
        )
        setContent {
            AkashicTheme {
                val attachmentPicker = rememberLauncherForActivityResult(
                    ActivityResultContracts.OpenMultipleDocuments(),
                ) { uris ->
                    if (uris.isNotEmpty()) viewModel.addAttachments(uris)
                }
                val session by viewModel.sessionState.collectAsStateWithLifecycle()
                val conversation by viewModel.conversationState.collectAsStateWithLifecycle()
                if (!session.initialized) {
                    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        CircularProgressIndicator()
                    }
                } else if (session.hasProfile) {
                    MobileConversationScaffold(
                        state = conversation,
                        onSelectSession = viewModel::selectSession,
                        onNewSession = viewModel::createSession,
                        onRestartPairing = viewModel::restartPairing,
                        onAttach = { attachmentPicker.launch(arrayOf("*/*")) },
                        onRemoveAttachment = viewModel::removeAttachment,
                        onRetryAttachment = viewModel::retryAttachment,
                        onSend = viewModel::sendMessage,
                        onStop = {},
                    )
                } else {
                    PairingScreen(
                        scanGeneration = session.scanGeneration,
                        confirmationCode = session.pairingConfirmationCode,
                        errorMessage = session.errorMessage,
                        onQrCode = viewModel::onQrCode,
                    )
                }
            }
        }
    }
}
