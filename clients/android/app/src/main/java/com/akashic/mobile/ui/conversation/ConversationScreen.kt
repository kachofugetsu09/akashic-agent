package com.akashic.mobile.ui.conversation

import android.animation.ValueAnimator
import androidx.activity.compose.BackHandler
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.animation.togetherWith
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
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.rounded.Menu
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.Send
import androidx.compose.material.icons.rounded.AttachFile
import androidx.compose.material.icons.rounded.Build
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.CloudUpload
import androidx.compose.material.icons.rounded.ErrorOutline
import androidx.compose.material.icons.rounded.KeyboardArrowDown
import androidx.compose.material.icons.rounded.Phonelink
import androidx.compose.material.icons.rounded.Refresh
import androidx.compose.material.icons.rounded.Stop
import androidx.compose.material.icons.rounded.StopCircle
import androidx.compose.material.icons.rounded.Sync
import androidx.compose.material.icons.rounded.Warning
import androidx.compose.material.icons.rounded.Wifi
import androidx.compose.material.icons.rounded.WifiOff
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonColors
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.BlurredEdgeTreatment
import androidx.compose.ui.draw.blur
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.semantics.ProgressBarRangeInfo
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.paneTitle
import androidx.compose.ui.semantics.progressBarRangeInfo
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.akashic.mobile.ui.design.AkashicTheme
import com.akashic.mobile.ui.design.pressScale
import com.hrm.latex.renderer.LatexAutoWrap
import com.hrm.latex.renderer.model.LatexConfig
import com.hrm.latex.renderer.model.LatexTheme
import com.mikepenz.markdown.m3.Markdown
import com.mikepenz.markdown.m3.markdownTypography
import com.mikepenz.markdown.model.rememberMarkdownState
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.distinctUntilChanged

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ConversationScreen(
    state: ConversationUiState,
    onAttach: () -> Unit,
    onRemoveAttachment: (String) -> Unit,
    onRetryAttachment: (String) -> Unit,
    onSend: (String) -> Unit,
    onSendCommand: (String) -> Unit,
    onStop: () -> Unit,
    onRetryDownloadedAttachment: (String) -> Unit,
    onOpenDownloadedAttachment: (String) -> Unit,
    onDismissError: () -> Unit,
) {
    var composerText by rememberSaveable { mutableStateOf("") }
    var commandPanelOpen by rememberSaveable { mutableStateOf(false) }
    var composerFocused by remember { mutableStateOf(false) }
    var restoreComposerAfterPanel by remember { mutableStateOf(false) }
    val composerFocusRequester = remember { FocusRequester() }
    val focusManager = LocalFocusManager.current
    val keyboardController = LocalSoftwareKeyboardController.current

    Scaffold(
        containerColor = MaterialTheme.colorScheme.surface,
        topBar = {
            ConversationTopBar(
                connectionLabel = state.connectionLabel,
                connectionStatus = state.connectionStatus,
            )
        },
        bottomBar = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(MaterialTheme.colorScheme.surface)
                    .testTag("command-composer-stack"),
            ) {
                AnimatedVisibility(
                    visible = commandPanelOpen,
                    enter = expandVertically(
                        animationSpec = tween(220),
                        expandFrom = Alignment.Bottom,
                    ) + fadeIn(tween(160)),
                    exit = shrinkVertically(
                        animationSpec = tween(180),
                        shrinkTowards = Alignment.Bottom,
                    ) + fadeOut(tween(120)),
                ) {
                    CommandPanel(
                        commands = state.commands,
                        onSelect = { command ->
                            if (commandPanelOpen) {
                                commandPanelOpen = false
                                restoreComposerAfterPanel = false
                                onSendCommand("/${command.command}")
                            }
                        },
                    )
                }
                ConversationBottomBar(
                    text = composerText,
                    onTextChange = { composerText = it },
                    connectionNotice = state.connectionNotice,
                    errorNotice = state.errorNotice,
                    isStreaming = state.isStreaming,
                    isStopping = state.isStopping,
                    canStop = state.canStop,
                    enabled = state.canSend,
                    attachments = state.attachments,
                    hasCommands = state.commands.isNotEmpty(),
                    commandPanelOpen = commandPanelOpen,
                    composerFocusRequester = composerFocusRequester,
                    onComposerFocusChanged = { focused ->
                        composerFocused = focused
                        if (focused && commandPanelOpen) {
                            commandPanelOpen = false
                            restoreComposerAfterPanel = false
                        }
                    },
                    onToggleCommands = {
                        if (commandPanelOpen) {
                            commandPanelOpen = false
                            if (restoreComposerAfterPanel) {
                                composerFocusRequester.requestFocus()
                                keyboardController?.show()
                            }
                        } else {
                            restoreComposerAfterPanel = composerFocused
                            commandPanelOpen = true
                            focusManager.clearFocus(force = true)
                            keyboardController?.hide()
                        }
                    },
                    onAttach = {
                        commandPanelOpen = false
                        restoreComposerAfterPanel = false
                        onAttach()
                    },
                    onRemoveAttachment = onRemoveAttachment,
                    onRetryAttachment = onRetryAttachment,
                    onSend = {
                        commandPanelOpen = false
                        onSend(composerText)
                        composerText = ""
                        focusManager.clearFocus(force = true)
                        keyboardController?.hide()
                    },
                    onStop = onStop,
                    onDismissError = onDismissError,
                )
            }
        },
    ) { contentPadding ->
        if (state.messages.isEmpty()) {
            EmptyConversation(Modifier.padding(contentPadding))
        } else {
            MessageList(
                messages = state.messages,
                onRetryDownloadedAttachment = onRetryDownloadedAttachment,
                onOpenDownloadedAttachment = onOpenDownloadedAttachment,
                modifier = Modifier.padding(contentPadding),
            )
        }
    }

    BackHandler(enabled = commandPanelOpen) {
        commandPanelOpen = false
        if (restoreComposerAfterPanel) {
            composerFocusRequester.requestFocus()
            keyboardController?.show()
        }
    }
    LaunchedEffect(state.canSend, state.commands.isEmpty()) {
        if (!state.canSend || state.commands.isEmpty()) commandPanelOpen = false
    }
}

@Composable
private fun CommandPanel(
    commands: List<CommandUi>,
    onSelect: (CommandUi) -> Unit,
) {
    Surface(
        shape = RoundedCornerShape(28.dp),
        color = MaterialTheme.colorScheme.surfaceContainerHighest,
        tonalElevation = 1.dp,
        shadowElevation = 2.dp,
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 240.dp, max = 304.dp)
            .padding(horizontal = 12.dp, vertical = 4.dp)
            .semantics { paneTitle = "快捷命令" }
            .testTag("command-panel"),
    ) {
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .testTag("command-panel-list"),
            contentPadding = PaddingValues(vertical = 8.dp),
        ) {
            items(commands, key = { it.command }) { command ->
                Surface(
                    onClick = { onSelect(command) },
                    color = Color.Transparent,
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 56.dp)
                        .testTag("command-${command.command}"),
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 20.dp, vertical = 10.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(16.dp),
                    ) {
                        Text(
                            text = command.description,
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurface,
                            modifier = Modifier.weight(1f),
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis,
                        )
                        Text(
                            text = "/${command.command}",
                            style = MaterialTheme.typography.bodyMedium,
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Medium,
                            color = MaterialTheme.colorScheme.primary,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ConversationTopBar(
    connectionLabel: String,
    connectionStatus: ConnectionStatusUi,
) {
    TopAppBar(
        title = {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(7.dp),
            ) {
                val icon = when (connectionStatus) {
                    ConnectionStatusUi.CONNECTING, ConnectionStatusUi.RECONNECTING -> Icons.Rounded.Sync
                    ConnectionStatusUi.READY -> Icons.Rounded.Wifi
                    ConnectionStatusUi.DEGRADED -> Icons.Rounded.Warning
                    ConnectionStatusUi.DISCONNECTED -> Icons.Rounded.WifiOff
                }
                val tint = when (connectionStatus) {
                    ConnectionStatusUi.READY -> MaterialTheme.colorScheme.primary
                    ConnectionStatusUi.DEGRADED -> MaterialTheme.colorScheme.tertiary
                    ConnectionStatusUi.DISCONNECTED -> MaterialTheme.colorScheme.error
                    ConnectionStatusUi.CONNECTING, ConnectionStatusUi.RECONNECTING -> {
                        MaterialTheme.colorScheme.onSurfaceVariant
                    }
                }
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = tint,
                    modifier = Modifier.size(18.dp),
                )
                Text(
                    text = connectionLabel,
                    style = MaterialTheme.typography.labelLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        },
        navigationIcon = {
            Spacer(Modifier.size(48.dp))
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
            imageVector = Icons.Rounded.Phonelink,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.primary,
            modifier = Modifier.size(44.dp),
        )
        Spacer(Modifier.height(20.dp))
        Text(
            text = "开始一段手机对话",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.onSurface,
        )
        Spacer(Modifier.height(8.dp))
        Text(
            text = "这里的会话与电脑 WebChat 分开保存。",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun MessageList(
    messages: List<MessageUi>,
    onRetryDownloadedAttachment: (String) -> Unit,
    onOpenDownloadedAttachment: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    val listState = rememberLazyListState()
    var followsBottom by remember { mutableStateOf(true) }
    val bottomThreshold = with(LocalDensity.current) { 48.dp.roundToPx() }
    val contentRevision = remember(messages) { messageContentRevision(messages) }

    LaunchedEffect(listState, bottomThreshold) {
        snapshotFlow {
            val layout = listState.layoutInfo
            val last = layout.visibleItemsInfo.lastOrNull()
            val atBottom = isConversationAtBottom(
                totalItems = layout.totalItemsCount,
                lastVisibleIndex = last?.index,
                lastVisibleEnd = last?.let { it.offset + it.size },
                viewportEnd = layout.viewportEndOffset,
                threshold = bottomThreshold,
            )
            listState.isScrollInProgress to atBottom
        }.collect { (isScrolling, atBottom) ->
            if (isScrolling) followsBottom = atBottom
            else if (atBottom) followsBottom = true
        }
    }

    LaunchedEffect(contentRevision) {
        if (followsBottom) listState.scrollToItem(messages.size)
    }

    LaunchedEffect(listState, messages.size) {
        snapshotFlow {
            val layout = listState.layoutInfo
            val last = layout.visibleItemsInfo.lastOrNull()
            listOf(
                layout.totalItemsCount,
                last?.index ?: -1,
                last?.offset ?: 0,
                last?.size ?: 0,
                layout.viewportEndOffset,
            )
        }.distinctUntilChanged().collect {
            if (followsBottom && !listState.isScrollInProgress && listState.layoutInfo.totalItemsCount > 0) {
                listState.scrollToItem(listState.layoutInfo.totalItemsCount - 1)
            }
        }
    }

    LazyColumn(
        modifier = modifier
            .fillMaxSize()
            .testTag("conversation-message-list"),
        userScrollEnabled = true,
        state = listState,
        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 20.dp),
    ) {
        itemsIndexed(messages, key = { _, message -> message.id }) { index, message ->
            Box(Modifier.testTag("message-${message.id}")) {
                Column {
                    if (index > 0) {
                        Spacer(Modifier.height(14.dp))
                        if (messages[index - 1]::class == message::class) {
                            HorizontalDivider(
                                color = MaterialTheme.colorScheme.outline.copy(alpha = 0.24f),
                            )
                        }
                        Spacer(Modifier.height(14.dp))
                    }
                    when (message) {
                        is MessageUi.User -> UserMessage(
                            message,
                            onRetryDownloadedAttachment,
                            onOpenDownloadedAttachment,
                        )
                        is MessageUi.AssistantTurn -> AssistantTurn(
                            message,
                            onRetryDownloadedAttachment,
                            onOpenDownloadedAttachment,
                        )
                    }
                }
            }
        }
        item(key = "message-list-bottom") {
            Spacer(Modifier.height(1.dp))
        }
    }
}

@Composable
private fun UserMessage(
    message: MessageUi.User,
    onRetryDownloadedAttachment: (String) -> Unit,
    onOpenDownloadedAttachment: (String) -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.End,
    ) {
        if (message.text.isNotBlank()) {
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
                MarkdownMessage(
                    content = message.text,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
                )
            }
        }
        if (message.attachments.isNotEmpty()) {
            MessageAttachments(
                attachments = message.attachments,
                onRetry = onRetryDownloadedAttachment,
                onOpen = onOpenDownloadedAttachment,
                modifier = Modifier
                    .fillMaxWidth(0.86f)
                    .padding(top = if (message.text.isBlank()) 0.dp else 8.dp),
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
private fun AssistantTurn(
    message: MessageUi.AssistantTurn,
    onRetryDownloadedAttachment: (String) -> Unit,
    onOpenDownloadedAttachment: (String) -> Unit,
) {
    Column {
        message.intro?.let { intro ->
            Text(
                text = intro,
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurface,
            )
            Spacer(Modifier.height(12.dp))
        }
        if (message.blocks.isNotEmpty()) {
            ProcessDisclosure(message)
        }
        if (message.answer.isNotBlank()) {
            MarkdownMessage(
                content = message.answer,
                modifier = Modifier.padding(top = 8.dp),
            )
        }
        if (message.attachments.isNotEmpty()) {
            MessageAttachments(
                attachments = message.attachments,
                onRetry = onRetryDownloadedAttachment,
                onOpen = onOpenDownloadedAttachment,
                modifier = Modifier.padding(top = 10.dp),
            )
        }
        if (message.status == AssistantTurnStatus.INTERRUPTED) {
            Row(
                modifier = Modifier
                    .padding(top = 12.dp)
                    .semantics { liveRegion = LiveRegionMode.Polite }
                    .testTag("turn-interrupted-${message.id}"),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(7.dp),
            ) {
                Icon(
                    imageVector = Icons.Rounded.StopCircle,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.tertiary,
                    modifier = Modifier.size(18.dp),
                )
                Text(
                    text = "生成已中止，可继续补充",
                    style = MaterialTheme.typography.labelLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun MarkdownMessage(content: String, modifier: Modifier = Modifier) {
    val segments = remember(content) { richMessageSegments(content) }
    val typography = markdownTypography(
        h1 = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.SemiBold),
        h2 = MaterialTheme.typography.titleLarge.copy(
            fontSize = 20.sp,
            lineHeight = 26.sp,
            fontWeight = FontWeight.SemiBold,
        ),
        h3 = MaterialTheme.typography.titleMedium.copy(
            fontSize = 18.sp,
            lineHeight = 24.sp,
            fontWeight = FontWeight.SemiBold,
        ),
        h4 = MaterialTheme.typography.bodyLarge.copy(fontWeight = FontWeight.SemiBold),
        h5 = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold),
        h6 = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.SemiBold),
        text = MaterialTheme.typography.bodyLarge,
        paragraph = MaterialTheme.typography.bodyLarge,
        ordered = MaterialTheme.typography.bodyLarge,
        bullet = MaterialTheme.typography.bodyLarge,
        list = MaterialTheme.typography.bodyLarge,
        table = MaterialTheme.typography.bodyMedium,
    )
    SelectionContainer(modifier = modifier) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            segments.forEachIndexed { index, segment ->
                when (segment) {
                    is RichMessageSegment.Markdown -> {
                        val markdownState = rememberMarkdownState(segment.content, retainState = true)
                        Markdown(
                            markdownState = markdownState,
                            typography = typography,
                            modifier = Modifier.fillMaxWidth(),
                        )
                    }
                    is RichMessageSegment.BlockMath -> LatexAutoWrap(
                        latex = segment.content,
                        config = LatexConfig(
                            fontSize = 18.sp,
                            theme = LatexTheme.material3(),
                            accessibilityEnabled = true,
                        ),
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 4.dp)
                            .testTag("message-math-$index"),
                    )
                }
            }
        }
    }
}

@Composable
private fun ProcessDisclosure(message: MessageUi.AssistantTurn) {
    var expanded by rememberSaveable(message.id) { mutableStateOf(message.isStreaming) }
    var hasStreamed by remember(message.id) { mutableStateOf(message.isStreaming) }
    val animationsEnabled = remember { ValueAnimator.areAnimatorsEnabled() }
    val easing = CubicBezierEasing(0.2f, 0f, 0f, 1f)
    val rotation by animateFloatAsState(
        targetValue = if (expanded) 180f else 0f,
        animationSpec = if (animationsEnabled) tween(300, easing = easing) else snap(),
        label = "thinking disclosure",
    )

    LaunchedEffect(message.isStreaming) {
        if (message.isStreaming) {
            hasStreamed = true
            expanded = true
        } else if (hasStreamed) {
            delay(1_000)
            expanded = false
            hasStreamed = false
        }
    }

    Surface(
        onClick = { if (!message.isStreaming) expanded = !expanded },
        color = Color.Transparent,
        contentColor = MaterialTheme.colorScheme.onSurfaceVariant,
        shape = RoundedCornerShape(22.dp),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .height(44.dp)
                .padding(horizontal = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Text(
                text = when (message.status) {
                    AssistantTurnStatus.STREAMING -> "正在思考"
                    AssistantTurnStatus.COMPLETE -> {
                        "已思考${message.durationSeconds?.let { " ${it}s" } ?: ""}"
                    }
                    AssistantTurnStatus.INTERRUPTED -> {
                        "已中止${message.durationSeconds?.let { " · ${it}s" } ?: ""}"
                    }
                },
                style = MaterialTheme.typography.labelLarge,
                modifier = Modifier.weight(1f),
            )
            Icon(
                imageVector = Icons.Rounded.KeyboardArrowDown,
                contentDescription = if (expanded) "收起思考过程" else "展开思考过程",
                modifier = Modifier
                    .size(20.dp)
                    .graphicsLayer { rotationZ = rotation },
            )
        }
    }

    AnimatedVisibility(
        visible = expanded,
        enter = expandVertically(
            animationSpec = if (animationsEnabled) tween(300, easing = easing) else snap(),
            expandFrom = Alignment.Top,
        ) + fadeIn(animationSpec = if (animationsEnabled) tween(180) else snap()),
        exit = shrinkVertically(
            animationSpec = if (animationsEnabled) tween(300, easing = easing) else snap(),
            shrinkTowards = Alignment.Top,
        ) + fadeOut(animationSpec = if (animationsEnabled) tween(140) else snap()),
    ) {
        Box(Modifier.padding(bottom = 16.dp)) {
            ProcessRail(message.blocks)
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
            )
        }
    }
}

@Composable
private fun ProcessStep(
    block: ProcessBlockUi,
    isFirst: Boolean,
) {
    val nodeColor = when (block.state) {
        ProcessBlockState.RUNNING -> MaterialTheme.colorScheme.tertiary
        ProcessBlockState.FAILED -> MaterialTheme.colorScheme.error
        ProcessBlockState.COMPLETED -> MaterialTheme.colorScheme.outline
    }
    val railColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.38f)
    val nodeRingColor = MaterialTheme.colorScheme.surface

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
            val nodeY = 14.dp.toPx()
            val lineStart = if (isFirst) nodeY else 0f
            val lineEnd = (size.height - 6.dp.toPx()).coerceAtLeast(nodeY)
            drawLine(railColor, Offset(center.x, lineStart), Offset(center.x, lineEnd), 1.dp.toPx())

            if (block.kind == ProcessBlockKind.THINKING) {
                drawCircle(nodeRingColor, radius = 7.dp.toPx(), center = Offset(center.x, nodeY))
                drawCircle(nodeColor, radius = 3.5.dp.toPx(), center = Offset(center.x, nodeY))
            } else {
                fun diamond(radius: Float, color: Color) {
                    val path = Path().apply {
                        moveTo(center.x, nodeY - radius)
                        lineTo(center.x + radius, nodeY)
                        lineTo(center.x, nodeY + radius)
                        lineTo(center.x - radius, nodeY)
                        close()
                    }
                    drawPath(path, color)
                }
                diamond(7.dp.toPx(), nodeRingColor)
                diamond(4.dp.toPx(), nodeColor)
            }
        }

        Column(
            modifier = Modifier
                .weight(1f)
                .padding(start = 8.dp, top = 4.dp, end = 8.dp, bottom = 12.dp),
            verticalArrangement = Arrangement.spacedBy(2.dp),
        ) {
            if (block.kind == ProcessBlockKind.THINKING) {
                Text(
                    text = block.detail,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            } else {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(7.dp),
                ) {
                    Icon(
                        imageVector = Icons.Rounded.Build,
                        contentDescription = null,
                        tint = if (block.state == ProcessBlockState.RUNNING) {
                            MaterialTheme.colorScheme.tertiary
                        } else {
                            MaterialTheme.colorScheme.onSurfaceVariant
                        },
                        modifier = Modifier.size(14.dp),
                    )
                    Text(
                        text = block.title,
                        style = MaterialTheme.typography.bodyMedium,
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Medium,
                        color = MaterialTheme.colorScheme.onSurface,
                    )
                }
                if (block.detail.isNotBlank()) {
                    Text(
                        text = block.detail,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

@Composable
private fun ConversationBottomBar(
    text: String,
    onTextChange: (String) -> Unit,
    connectionNotice: String?,
    errorNotice: String?,
    isStreaming: Boolean,
    isStopping: Boolean,
    canStop: Boolean,
    enabled: Boolean,
    attachments: List<ComposerAttachmentUi>,
    hasCommands: Boolean,
    commandPanelOpen: Boolean,
    composerFocusRequester: FocusRequester,
    onComposerFocusChanged: (Boolean) -> Unit,
    onToggleCommands: () -> Unit,
    onAttach: () -> Unit,
    onRemoveAttachment: (String) -> Unit,
    onRetryAttachment: (String) -> Unit,
    onSend: () -> Unit,
    onStop: () -> Unit,
    onDismissError: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surface)
            .windowInsetsPadding(WindowInsets.navigationBars)
            .imePadding()
            .padding(start = 12.dp, end = 12.dp, bottom = 10.dp),
    ) {
        if (errorNotice != null) {
            DismissibleErrorNotice(
                message = errorNotice,
                onDismiss = onDismissError,
                modifier = Modifier.padding(bottom = 8.dp),
            )
        }
        if (connectionNotice != null) {
            Text(
                text = connectionNotice,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(start = 12.dp, bottom = 8.dp),
            )
        }
        AnimatedVisibility(
            visible = isStopping,
            enter = fadeIn(tween(120)),
            exit = fadeOut(tween(120)),
        ) {
            Text(
                text = "正在中止本轮…",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.tertiary,
                modifier = Modifier
                    .padding(start = 12.dp, bottom = 8.dp)
                    .semantics { liveRegion = LiveRegionMode.Polite }
                    .testTag("turn-stop-pending"),
            )
        }
        AnimatedVisibility(
            visible = attachments.isNotEmpty(),
            enter = expandVertically(animationSpec = tween(220)) + fadeIn(animationSpec = tween(180)),
            exit = shrinkVertically(animationSpec = tween(180)) + fadeOut(animationSpec = tween(140)),
        ) {
            AttachmentDraftStrip(
                attachments = attachments,
                onRemove = onRemoveAttachment,
                onRetry = onRetryAttachment,
                modifier = Modifier.padding(bottom = 10.dp),
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
            TactileIconButton(
                onClick = onToggleCommands,
                enabled = enabled && hasCommands,
                colors = if (commandPanelOpen) {
                    IconButtonDefaults.filledIconButtonColors(
                        containerColor = MaterialTheme.colorScheme.primary,
                        contentColor = MaterialTheme.colorScheme.onPrimary,
                    )
                } else {
                    IconButtonDefaults.iconButtonColors()
                },
            ) {
                Icon(
                    imageVector = if (commandPanelOpen) Icons.Rounded.Close else Icons.Rounded.Menu,
                    contentDescription = if (commandPanelOpen) "关闭快捷命令" else "打开快捷命令",
                    modifier = Modifier.testTag("composer-command-menu"),
                )
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
                    .focusRequester(composerFocusRequester)
                    .onFocusChanged { onComposerFocusChanged(it.isFocused) }
                    .testTag("composer-input")
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
            TactileIconButton(onClick = onAttach, enabled = enabled) {
                Icon(
                    Icons.Rounded.AttachFile,
                    contentDescription = "添加附件",
                    modifier = Modifier.testTag("composer-attachment"),
                )
            }
            SendStopButton(
                showStop = isStreaming,
                isStopping = isStopping,
                enabled = if (isStreaming) {
                    canStop
                } else {
                    enabled &&
                        (text.isNotBlank() || attachments.isNotEmpty()) &&
                        attachments.all { it.state == ComposerAttachmentState.READY }
                },
                onClick = if (isStreaming) onStop else onSend,
            )
        }
    }
}

@Composable
private fun DismissibleErrorNotice(
    message: String,
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Surface(
        color = MaterialTheme.colorScheme.errorContainer,
        contentColor = MaterialTheme.colorScheme.onErrorContainer,
        shape = MaterialTheme.shapes.medium,
        modifier = modifier
            .fillMaxWidth()
            .semantics { liveRegion = LiveRegionMode.Assertive }
            .testTag("conversation-error-notice"),
    ) {
        Row(
            modifier = Modifier.padding(start = 14.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                Icons.Rounded.ErrorOutline,
                contentDescription = null,
                modifier = Modifier.size(20.dp),
            )
            Text(
                text = message,
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier
                    .weight(1f)
                    .padding(horizontal = 10.dp, vertical = 12.dp),
            )
            TactileIconButton(onClick = onDismiss) {
                Icon(Icons.Rounded.Close, contentDescription = "关闭错误提示")
            }
        }
    }
}

@Composable
private fun AttachmentDraftStrip(
    attachments: List<ComposerAttachmentUi>,
    onRemove: (String) -> Unit,
    onRetry: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    LazyRow(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        contentPadding = PaddingValues(horizontal = 2.dp),
    ) {
        items(attachments, key = { it.id }) { attachment ->
            AttachmentDraftItem(attachment, onRemove, onRetry)
        }
    }
}

@Composable
private fun AttachmentDraftItem(
    attachment: ComposerAttachmentUi,
    onRemove: (String) -> Unit,
    onRetry: (String) -> Unit,
) {
    val progress = attachment.transferredBytes.toFloat() / attachment.sizeBytes.toFloat()
    val animationsEnabled = remember { ValueAnimator.areAnimatorsEnabled() }
    val animatedProgress by animateFloatAsState(
        targetValue = progress,
        animationSpec = if (animationsEnabled) tween(160) else snap(),
        label = "attachment progress",
    )
    val stateLabel = when (attachment.state) {
        ComposerAttachmentState.WAITING_FOR_CONNECTION -> "等待网络"
        ComposerAttachmentState.UPLOADING -> "上传中 ${(animatedProgress * 100).toInt()}%"
        ComposerAttachmentState.READY -> "已就绪"
        ComposerAttachmentState.FAILED -> "上传失败"
    }
    Surface(
        color = MaterialTheme.colorScheme.surfaceContainerLow,
        contentColor = MaterialTheme.colorScheme.onSurface,
        shape = RoundedCornerShape(20.dp),
        modifier = Modifier
            .width(280.dp)
            .testTag("attachment-draft-${attachment.id}")
            .semantics {
                stateDescription = "文件 ${attachment.filename}，$stateLabel"
                if (attachment.state == ComposerAttachmentState.UPLOADING) {
                    progressBarRangeInfo = ProgressBarRangeInfo(animatedProgress, 0f..1f)
                }
            },
    ) {
        Row(
            modifier = Modifier.padding(start = 14.dp, top = 10.dp, bottom = 10.dp, end = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            AnimatedContent(
                targetState = attachment.state,
                transitionSpec = {
                    (fadeIn(tween(180)) + scaleIn(tween(220), initialScale = 0.25f)) togetherWith
                        (fadeOut(tween(140)) + scaleOut(tween(180), targetScale = 0.25f))
                },
                label = "attachment state",
            ) { state ->
                val icon = when (state) {
                    ComposerAttachmentState.WAITING_FOR_CONNECTION -> Icons.Rounded.CloudUpload
                    ComposerAttachmentState.UPLOADING -> Icons.Rounded.CloudUpload
                    ComposerAttachmentState.READY -> Icons.Rounded.CheckCircle
                    ComposerAttachmentState.FAILED -> Icons.Rounded.ErrorOutline
                }
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = when (state) {
                        ComposerAttachmentState.FAILED -> MaterialTheme.colorScheme.error
                        ComposerAttachmentState.READY -> MaterialTheme.colorScheme.primary
                        else -> MaterialTheme.colorScheme.primary
                    },
                    modifier = Modifier.size(22.dp),
                )
            }
            Column(
                modifier = Modifier
                    .weight(1f)
                    .padding(start = 10.dp),
            ) {
                Text(
                    text = attachment.filename,
                    style = MaterialTheme.typography.labelLarge,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Text(
                    text = "${formatFileSize(attachment.sizeBytes)} · $stateLabel",
                    style = MaterialTheme.typography.labelMedium.merge(
                        TextStyle(fontFeatureSettings = "tnum"),
                    ),
                    color = if (attachment.state == ComposerAttachmentState.FAILED) {
                        MaterialTheme.colorScheme.error
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant
                    },
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                if (attachment.state == ComposerAttachmentState.UPLOADING) {
                    LinearProgressIndicator(
                        progress = { animatedProgress },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 6.dp),
                        color = MaterialTheme.colorScheme.primary,
                        trackColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.16f),
                    )
                }
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (attachment.state == ComposerAttachmentState.FAILED) {
                    TactileIconButton(onClick = { onRetry(attachment.id) }) {
                        Icon(Icons.Rounded.Refresh, contentDescription = "重试上传 ${attachment.filename}")
                    }
                }
                if (attachment.canRemove) {
                    TactileIconButton(onClick = { onRemove(attachment.id) }) {
                        Icon(Icons.Rounded.Close, contentDescription = "移除附件 ${attachment.filename}")
                    }
                }
                if (attachment.state in setOf(
                        ComposerAttachmentState.WAITING_FOR_CONNECTION,
                        ComposerAttachmentState.UPLOADING,
                    )
                ) {
                    Spacer(Modifier.width(8.dp))
                }
            }
        }
    }
}

private fun formatFileSize(bytes: Long): String = when {
    bytes >= 1024 * 1024 -> "%.1f MB".format(bytes / (1024.0 * 1024.0))
    bytes >= 1024 -> "%.1f KB".format(bytes / 1024.0)
    else -> "$bytes B"
}

@Composable
private fun TactileIconButton(
    onClick: () -> Unit,
    enabled: Boolean = true,
    colors: IconButtonColors = IconButtonDefaults.iconButtonColors(),
    content: @Composable () -> Unit,
) {
    val interactionSource = remember { MutableInteractionSource() }
    IconButton(
        onClick = onClick,
        enabled = enabled,
        interactionSource = interactionSource,
        colors = colors,
        modifier = Modifier
            .size(48.dp)
            .pressScale(interactionSource, enabled),
        content = content,
    )
}

@Composable
private fun SendStopButton(
    showStop: Boolean,
    isStopping: Boolean,
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
            .testTag("composer-send-stop")
            .semantics {
                contentDescription = when {
                    isStopping -> "正在停止生成"
                    showStop -> "停止生成"
                    else -> "发送消息"
                }
            }
            .pressScale(interactionSource, enabled),
    ) {
        if (isStopping) {
            androidx.compose.material3.CircularProgressIndicator(
                color = MaterialTheme.colorScheme.onPrimary,
                strokeWidth = 2.dp,
                modifier = Modifier
                    .size(22.dp)
                    .semantics { stateDescription = "正在停止生成" },
            )
        } else {
            ContextualSendStopIcon(showStop)
        }
    }
}

internal fun isConversationAtBottom(
    totalItems: Int,
    lastVisibleIndex: Int?,
    lastVisibleEnd: Int?,
    viewportEnd: Int,
    threshold: Int,
): Boolean = lastVisibleIndex == null || (
    lastVisibleIndex == totalItems - 1 &&
        requireNotNull(lastVisibleEnd) <= viewportEnd + threshold
    )

internal fun messageContentRevision(messages: List<MessageUi>): Int = messages.fold(1) { hash, message ->
    val messageHash = when (message) {
        is MessageUi.User -> listOf(
            message.id,
            message.text,
            message.deliveryLabel,
            message.attachments.joinToString { "${it.id}:${it.state}:${it.transferredBytes}" },
        ).hashCode()
        is MessageUi.AssistantTurn -> listOf(
            message.id,
            message.intro,
            message.answer,
            message.isStreaming,
            message.blocks.joinToString { "${it.id}:${it.state}:${it.detail}" },
            message.attachments.joinToString { "${it.id}:${it.state}:${it.transferredBytes}" },
        ).hashCode()
    }
    31 * hash + messageHash
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
            contentDescription = null,
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
            contentDescription = null,
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
        MobileConversationScaffold(
            state = PreviewConversationState,
            onSelectSession = {},
            onNewSession = {},
            onAttach = {},
            onRemoveAttachment = {},
            onRetryAttachment = {},
            onRetryDownloadedAttachment = {},
            onOpenDownloadedAttachment = {},
            onDismissError = {},
            onSend = {},
            onSendCommand = {},
            onStop = {},
        )
    }
}

@Preview(name = "Conversation dark", showBackground = true, widthDp = 412, heightDp = 860)
@Composable
private fun ConversationDarkPreview() {
    AkashicTheme(darkTheme = true) {
        MobileConversationScaffold(
            state = PreviewConversationState,
            onSelectSession = {},
            onNewSession = {},
            onAttach = {},
            onRemoveAttachment = {},
            onRetryAttachment = {},
            onRetryDownloadedAttachment = {},
            onOpenDownloadedAttachment = {},
            onDismissError = {},
            onSend = {},
            onSendCommand = {},
            onStop = {},
        )
    }
}
