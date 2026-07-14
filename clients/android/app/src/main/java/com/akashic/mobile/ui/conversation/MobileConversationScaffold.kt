package com.akashic.mobile.ui.conversation

import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.Edit
import androidx.compose.material.icons.rounded.Menu
import androidx.compose.material.icons.rounded.QrCodeScanner
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.NavigationDrawerItem
import androidx.compose.material3.NavigationDrawerItemDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.zIndex
import com.akashic.mobile.ui.design.pressScale
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MobileConversationScaffold(
    state: ConversationUiState,
    onSelectSession: (String) -> Unit,
    onNewSession: () -> Unit,
    onRestartPairing: () -> Unit = {},
    onAttach: () -> Unit,
    onRemoveAttachment: (String) -> Unit,
    onRetryAttachment: (String) -> Unit,
    onSend: (String) -> Unit,
    onStop: () -> Unit,
    onRetryDownloadedAttachment: (String) -> Unit,
    onOpenDownloadedAttachment: (String) -> Unit,
) {
    val drawerState = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()

    Box(Modifier.fillMaxSize()) {
        ModalNavigationDrawer(
            drawerState = drawerState,
            drawerContent = {
                MobileSessionDrawer(
                    state = state,
                    onSelectSession = { sessionId ->
                        onSelectSession(sessionId)
                        scope.launch { drawerState.close() }
                    },
                    onNewSession = {
                        onNewSession()
                        scope.launch { drawerState.close() }
                    },
                    onRestartPairing = {
                        onRestartPairing()
                        scope.launch { drawerState.close() }
                    },
                )
            },
            scrimColor = MaterialTheme.colorScheme.scrim.copy(alpha = 0.56f),
            modifier = Modifier.fillMaxSize(),
        ) {
            ConversationScreen(
                state = state,
                onAttach = onAttach,
                onRemoveAttachment = onRemoveAttachment,
                onRetryAttachment = onRetryAttachment,
                onSend = onSend,
                onStop = onStop,
                onRetryDownloadedAttachment = onRetryDownloadedAttachment,
                onOpenDownloadedAttachment = onOpenDownloadedAttachment,
            )
        }

        PersistentDrawerButton(
            isOpen = drawerState.isOpen,
            onClick = {
                scope.launch {
                    if (drawerState.isOpen) drawerState.close() else drawerState.open()
                }
            },
            modifier = Modifier
                .windowInsetsPadding(WindowInsets.statusBars)
                .padding(start = 4.dp, top = 4.dp)
                .zIndex(2f),
        )
    }
}

@Composable
private fun MobileSessionDrawer(
    state: ConversationUiState,
    onSelectSession: (String) -> Unit,
    onNewSession: () -> Unit,
    onRestartPairing: () -> Unit,
) {
    ModalDrawerSheet(
        modifier = Modifier
            .width(336.dp)
            .fillMaxHeight(),
        drawerShape = RoundedCornerShape(topEnd = 28.dp, bottomEnd = 28.dp),
        drawerContainerColor = MaterialTheme.colorScheme.surface,
    ) {
        Column(
            modifier = Modifier
                .fillMaxHeight()
                .windowInsetsPadding(WindowInsets.statusBars)
                .padding(bottom = 16.dp),
        ) {
            Column(
                modifier = Modifier
                    .height(76.dp)
                    .padding(start = 64.dp, top = 9.dp, end = 20.dp),
                verticalArrangement = Arrangement.Center,
            ) {
                Text(
                    text = "手机对话",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Medium,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                Text(
                    text = "Android 独立会话",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            Text(
                text = "最近",
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(start = 28.dp, top = 14.dp, bottom = 8.dp),
            )
            LazyColumn(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 12.dp),
                verticalArrangement = Arrangement.spacedBy(2.dp),
            ) {
                items(state.sessions, key = { it.sessionId }) { session ->
                    NavigationDrawerItem(
                        label = {
                            Text(
                                text = session.title,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                            )
                        },
                        selected = session.sessionId == state.selectedSessionId,
                        onClick = { onSelectSession(session.sessionId) },
                        shape = RoundedCornerShape(28.dp),
                        colors = NavigationDrawerItemDefaults.colors(
                            selectedContainerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
                            selectedTextColor = MaterialTheme.colorScheme.onSurface,
                            unselectedContainerColor = MaterialTheme.colorScheme.surface,
                            unselectedTextColor = MaterialTheme.colorScheme.onSurface,
                        ),
                    )
                }
            }

            NavigationDrawerItem(
                icon = {
                    Icon(
                        Icons.Rounded.QrCodeScanner,
                        contentDescription = null,
                        modifier = Modifier.size(20.dp),
                    )
                },
                label = { Text("重新扫码连接") },
                selected = false,
                onClick = onRestartPairing,
                shape = RoundedCornerShape(28.dp),
                colors = NavigationDrawerItemDefaults.colors(
                    unselectedContainerColor = MaterialTheme.colorScheme.surface,
                    unselectedTextColor = MaterialTheme.colorScheme.onSurfaceVariant,
                    unselectedIconColor = MaterialTheme.colorScheme.onSurfaceVariant,
                ),
                modifier = Modifier.padding(horizontal = 12.dp),
            )

            FilledTonalButton(
                onClick = onNewSession,
                shape = RoundedCornerShape(24.dp),
                colors = ButtonDefaults.filledTonalButtonColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    contentColor = MaterialTheme.colorScheme.onPrimaryContainer,
                ),
                modifier = Modifier
                    .padding(start = 16.dp, top = 12.dp)
                    .height(48.dp),
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Icon(Icons.Rounded.Edit, contentDescription = null, modifier = Modifier.size(20.dp))
                    Text("聊天", style = MaterialTheme.typography.labelLarge)
                }
            }
        }
    }
}

@Composable
private fun PersistentDrawerButton(
    isOpen: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val interactionSource = remember { MutableInteractionSource() }
    IconButton(
        onClick = onClick,
        interactionSource = interactionSource,
        modifier = modifier
            .size(48.dp)
            .pressScale(interactionSource),
    ) {
        Icon(
            imageVector = if (isOpen) Icons.Rounded.Close else Icons.Rounded.Menu,
            contentDescription = if (isOpen) "收起对话列表" else "打开对话列表",
            tint = MaterialTheme.colorScheme.onSurface,
        )
    }
}
