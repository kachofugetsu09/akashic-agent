package com.akashic.mobile.ui.conversation

import android.animation.ValueAnimator
import androidx.compose.animation.core.CubicBezierEasing
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.snap
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.Send
import androidx.compose.material.icons.rounded.AttachFile
import androidx.compose.material.icons.rounded.Check
import androidx.compose.material.icons.rounded.Lock
import androidx.compose.material.icons.rounded.Menu
import androidx.compose.material.icons.rounded.MoreVert
import androidx.compose.material.icons.rounded.PhonelinkLock
import androidx.compose.material.icons.rounded.Stop
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.BlurredEdgeTreatment
import androidx.compose.ui.draw.blur
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.akashic.mobile.ui.design.AkashicTheme
import com.akashic.mobile.ui.design.pressScale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ConversationScreen(
    state: ConversationUiState,
    onOpenNavigation: (() -> Unit)?,
    onOpenMenu: (() -> Unit)?,
    onAttach: () -> Unit,
    onSend: (String) -> Unit,
    onStop: () -> Unit,
) {
    var composerText by rememberSaveable { mutableStateOf("") }

    Scaffold(
        containerColor = MaterialTheme.colorScheme.surface,
        topBar = {
            ConversationTopBar(
                title = state.title,
                connectionLabel = state.connectionLabel,
                onOpenNavigation = onOpenNavigation,
                onOpenMenu = onOpenMenu,
            )
        },
        bottomBar = {
            ConversationBottomBar(
                text = composerText,
                onTextChange = { composerText = it },
                isConnectionDegraded = state.isConnectionDegraded,
                isStreaming = state.isStreaming,
                enabled = state.canSend,
                onAttach = onAttach,
                onSend = {
                    onSend(composerText)
                    composerText = ""
                },
                onStop = onStop,
            )
        },
    ) { contentPadding ->
        if (state.messages.isEmpty()) {
            EmptyConversation(Modifier.padding(contentPadding))
        } else {
            MessageList(
                messages = state.messages,
                modifier = Modifier.padding(contentPadding),
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ConversationTopBar(
    title: String,
    connectionLabel: String,
    onOpenNavigation: (() -> Unit)?,
    onOpenMenu: (() -> Unit)?,
) {
    TopAppBar(
        title = {
            Column(verticalArrangement = Arrangement.spacedBy(1.dp)) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    Icon(
                        imageVector = Icons.Rounded.Lock,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.size(12.dp),
                    )
                    Text(
                        text = connectionLabel,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        },
        navigationIcon = {
            if (onOpenNavigation != null) {
                TactileIconButton(onClick = onOpenNavigation) {
                    Icon(Icons.Rounded.Menu, contentDescription = "打开对话列表")
                }
            }
        },
        actions = {
            if (onOpenMenu != null) {
                TactileIconButton(onClick = onOpenMenu) {
                    Icon(Icons.Rounded.MoreVert, contentDescription = "更多选项")
                }
            }
        },
        colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.surface),
    )
}

@Composable
private fun EmptyConversation(modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(horizontal = 32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Icon(
            imageVector = Icons.Rounded.PhonelinkLock,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.primary,
            modifier = Modifier.size(44.dp),
        )
        Spacer(Modifier.height(20.dp))
        Text(
            text = "连接电脑后开始对话",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.onSurface,
        )
        Spacer(Modifier.height(8.dp))
        Text(
            text = "首次扫码并确认后，设备会安全记住这台 Akashic。",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun MessageList(
    messages: List<MessageUi>,
    modifier: Modifier = Modifier,
) {
    LazyColumn(
        modifier = modifier.fillMaxSize(),
        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 20.dp),
        verticalArrangement = Arrangement.spacedBy(28.dp),
    ) {
        items(messages, key = { message -> message.id }) { message ->
            when (message) {
                is MessageUi.User -> UserMessage(message)
                is MessageUi.AssistantTurn -> AssistantTurn(message)
            }
        }
    }
}

@Composable
private fun UserMessage(message: MessageUi.User) {
    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.End,
    ) {
        Surface(
            color = MaterialTheme.colorScheme.primaryContainer,
            contentColor = MaterialTheme.colorScheme.onPrimaryContainer,
            shape = RoundedCornerShape(
                topStart = 20.dp,
                topEnd = 4.dp,
                bottomEnd = 20.dp,
                bottomStart = 20.dp,
            ),
            modifier = Modifier.fillMaxWidth(0.86f),
        ) {
            Text(
                text = message.text,
                style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
            )
        }
        Text(
            text = message.deliveryLabel,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 5.dp, end = 4.dp),
        )
    }
}

@Composable
private fun AssistantTurn(message: MessageUi.AssistantTurn) {
    Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
        message.intro?.let { intro ->
            Text(
                text = intro,
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurface,
            )
        }
        if (message.blocks.isNotEmpty()) {
            ProcessRail(message.blocks)
        }
        if (message.answer.isNotBlank()) {
            Text(
                text = message.answer,
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurface,
            )
        }
    }
}

@Composable
private fun ProcessRail(blocks: List<ProcessBlockUi>) {
    Column(modifier = Modifier.fillMaxWidth()) {
        blocks.forEachIndexed { index, block ->
            ProcessStep(
                block = block,
                isFirst = index == 0,
                isLast = index == blocks.lastIndex,
            )
        }
    }
}

@Composable
private fun ProcessStep(
    block: ProcessBlockUi,
    isFirst: Boolean,
    isLast: Boolean,
) {
    val nodeColor = when (block.state) {
        ProcessBlockState.RUNNING -> MaterialTheme.colorScheme.tertiary
        ProcessBlockState.FAILED -> MaterialTheme.colorScheme.error
        ProcessBlockState.COMPLETED -> MaterialTheme.colorScheme.outline
    }
    val railColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.38f)

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .height(IntrinsicSize.Min),
    ) {
        Canvas(
            modifier = Modifier
                .width(28.dp)
                .fillMaxHeight(),
        ) {
            val nodeY = 18.dp.toPx()
            if (!isFirst) drawLine(railColor, Offset(center.x, 0f), Offset(center.x, nodeY), 1.dp.toPx())
            if (!isLast) drawLine(railColor, Offset(center.x, nodeY), Offset(center.x, size.height), 1.dp.toPx())
            drawCircle(nodeColor.copy(alpha = 0.16f), radius = 8.dp.toPx(), center = Offset(center.x, nodeY))
            drawCircle(nodeColor, radius = 3.5.dp.toPx(), center = Offset(center.x, nodeY))
        }

        val stepBackground = if (block.state == ProcessBlockState.RUNNING) {
            MaterialTheme.colorScheme.tertiaryContainer
        } else {
            Color.Transparent
        }
        val stepContent = if (block.state == ProcessBlockState.RUNNING) {
            MaterialTheme.colorScheme.onTertiaryContainer
        } else {
            MaterialTheme.colorScheme.onSurface
        }
        Column(
            modifier = Modifier
                .weight(1f)
                .background(stepBackground, RoundedCornerShape(12.dp))
                .padding(start = 12.dp, top = 10.dp, end = 12.dp, bottom = 12.dp),
            verticalArrangement = Arrangement.spacedBy(3.dp),
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                Text(
                    text = if (block.kind == ProcessBlockKind.THINKING) "思考" else "工具",
                    style = MaterialTheme.typography.labelMedium,
                    color = if (block.state == ProcessBlockState.RUNNING) {
                        MaterialTheme.colorScheme.tertiary
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant
                    },
                )
                if (block.state == ProcessBlockState.COMPLETED) {
                    Icon(
                        imageVector = Icons.Rounded.Check,
                        contentDescription = "已完成",
                        tint = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.size(14.dp),
                    )
                }
            }
            Text(
                text = block.title,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium,
                color = stepContent,
            )
            Text(
                text = block.detail,
                style = MaterialTheme.typography.bodySmall,
                color = if (block.state == ProcessBlockState.RUNNING) {
                    MaterialTheme.colorScheme.onTertiaryContainer.copy(alpha = 0.78f)
                } else {
                    MaterialTheme.colorScheme.onSurfaceVariant
                },
            )
        }
    }
}

@Composable
private fun ConversationBottomBar(
    text: String,
    onTextChange: (String) -> Unit,
    isConnectionDegraded: Boolean,
    isStreaming: Boolean,
    enabled: Boolean,
    onAttach: () -> Unit,
    onSend: () -> Unit,
    onStop: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surface)
            .windowInsetsPadding(WindowInsets.navigationBars)
            .imePadding()
            .padding(start = 12.dp, end = 12.dp, bottom = 10.dp),
    ) {
        if (isConnectionDegraded) {
            Text(
                text = "网络不稳 · 消息已缓存，正在续传",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(start = 12.dp, bottom = 8.dp),
            )
        }
        HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(alpha = 0.12f))
        Spacer(Modifier.height(10.dp))
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 56.dp)
                .background(
                    color = MaterialTheme.colorScheme.surfaceContainerHigh,
                    shape = RoundedCornerShape(28.dp),
                )
                .padding(4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            TactileIconButton(onClick = onAttach, enabled = enabled) {
                Icon(Icons.Rounded.AttachFile, contentDescription = "添加附件")
            }
            BasicTextField(
                value = text,
                onValueChange = onTextChange,
                enabled = enabled,
                textStyle = MaterialTheme.typography.bodyLarge.merge(
                    TextStyle(color = MaterialTheme.colorScheme.onSurface),
                ),
                cursorBrush = SolidColor(MaterialTheme.colorScheme.primary),
                modifier = Modifier
                    .weight(1f)
                    .padding(horizontal = 8.dp, vertical = 10.dp),
                decorationBox = { innerTextField ->
                    Box(contentAlignment = Alignment.CenterStart) {
                        if (text.isEmpty()) {
                            Text(
                                text = if (enabled) "输入消息" else "连接后即可发送消息",
                                style = MaterialTheme.typography.bodyLarge,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                        innerTextField()
                    }
                },
            )
            SendStopButton(
                showStop = isStreaming,
                enabled = isStreaming || (enabled && text.isNotBlank()),
                onClick = if (isStreaming) onStop else onSend,
            )
        }
    }
}

@Composable
private fun TactileIconButton(
    onClick: () -> Unit,
    enabled: Boolean = true,
    content: @Composable () -> Unit,
) {
    val interactionSource = remember { MutableInteractionSource() }
    IconButton(
        onClick = onClick,
        enabled = enabled,
        interactionSource = interactionSource,
        modifier = Modifier
            .size(48.dp)
            .pressScale(interactionSource, enabled),
        content = content,
    )
}

@Composable
private fun SendStopButton(
    showStop: Boolean,
    enabled: Boolean,
    onClick: () -> Unit,
) {
    val interactionSource = remember { MutableInteractionSource() }
    IconButton(
        onClick = onClick,
        enabled = enabled,
        interactionSource = interactionSource,
        colors = IconButtonDefaults.filledIconButtonColors(
            containerColor = MaterialTheme.colorScheme.primary,
            contentColor = MaterialTheme.colorScheme.onPrimary,
        ),
        modifier = Modifier
            .size(48.dp)
            .pressScale(interactionSource, enabled),
    ) {
        ContextualSendStopIcon(showStop)
    }
}

@Composable
private fun ContextualSendStopIcon(showStop: Boolean) {
    val animationsEnabled = remember { ValueAnimator.areAnimatorsEnabled() }
    val easing = CubicBezierEasing(0.2f, 0f, 0f, 1f)
    val floatSpec = if (animationsEnabled) tween<Float>(300, easing = easing) else snap()
    val dpSpec = if (animationsEnabled) tween<androidx.compose.ui.unit.Dp>(300, easing = easing) else snap()
    val stopProgress by animateFloatAsState(
        targetValue = if (showStop) 1f else 0f,
        animationSpec = floatSpec,
        label = "stop icon",
    )
    val sendProgress by animateFloatAsState(
        targetValue = if (showStop) 0f else 1f,
        animationSpec = floatSpec,
        label = "send icon",
    )
    val stopBlur by animateDpAsState(
        targetValue = if (showStop) 0.dp else 4.dp,
        animationSpec = dpSpec,
        label = "stop blur",
    )
    val sendBlur by animateDpAsState(
        targetValue = if (showStop) 4.dp else 0.dp,
        animationSpec = dpSpec,
        label = "send blur",
    )

    Box(contentAlignment = Alignment.Center) {
        Icon(
            imageVector = Icons.Rounded.Stop,
            contentDescription = "停止生成",
            modifier = Modifier
                .graphicsLayer {
                    alpha = stopProgress
                    scaleX = 0.25f + 0.75f * stopProgress
                    scaleY = 0.25f + 0.75f * stopProgress
                }
                .blur(stopBlur, BlurredEdgeTreatment.Unbounded),
        )
        Icon(
            imageVector = Icons.AutoMirrored.Rounded.Send,
            contentDescription = "发送消息",
            modifier = Modifier
                .graphicsLayer {
                    alpha = sendProgress
                    scaleX = 0.25f + 0.75f * sendProgress
                    scaleY = 0.25f + 0.75f * sendProgress
                }
                .blur(sendBlur, BlurredEdgeTreatment.Unbounded),
        )
    }
}

@Preview(name = "Conversation light", showBackground = true, widthDp = 412, heightDp = 860)
@Composable
private fun ConversationLightPreview() {
    AkashicTheme(darkTheme = false) {
        ConversationScreen(
            state = PreviewConversationState,
            onOpenNavigation = {},
            onOpenMenu = {},
            onAttach = {},
            onSend = {},
            onStop = {},
        )
    }
}

@Preview(name = "Conversation dark", showBackground = true, widthDp = 412, heightDp = 860)
@Composable
private fun ConversationDarkPreview() {
    AkashicTheme(darkTheme = true) {
        ConversationScreen(
            state = PreviewConversationState,
            onOpenNavigation = {},
            onOpenMenu = {},
            onAttach = {},
            onSend = {},
            onStop = {},
        )
    }
}
