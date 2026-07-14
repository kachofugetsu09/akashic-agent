package com.akashic.mobile.ui.conversation

import android.content.ActivityNotFoundException
import android.content.ClipData
import android.content.Context
import android.content.Intent
import android.widget.Toast
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.OpenInNew
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.Description
import androidx.compose.material.icons.rounded.Download
import androidx.compose.material.icons.rounded.ErrorOutline
import androidx.compose.material.icons.rounded.Refresh
import androidx.compose.material.icons.rounded.Share
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.ProgressBarRangeInfo
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.progressBarRangeInfo
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.core.content.FileProvider
import coil3.compose.AsyncImage
import coil3.request.ImageRequest
import coil3.request.crossfade
import com.akashic.mobile.BuildConfig
import com.akashic.mobile.ui.design.pressScale
import java.io.File
import kotlin.math.roundToInt

@Composable
internal fun MessageAttachments(
    attachments: List<MessageAttachmentUi>,
    onRetry: (String) -> Unit,
    onOpen: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    if (attachments.isEmpty()) return
    var preview by remember { mutableStateOf<MessageAttachmentUi?>(null) }
    var failedPreviews by remember(attachments.map { it.id to it.state }) {
        mutableStateOf(emptySet<String>())
    }
    val context = LocalContext.current

    Column(modifier = modifier.fillMaxWidth()) {
        attachments.forEachIndexed { index, attachment ->
            if (index > 0) {
                HorizontalDivider(
                    color = MaterialTheme.colorScheme.outline.copy(alpha = 0.16f),
                    modifier = Modifier.padding(vertical = 6.dp),
                )
            }
            if (
                attachment.isDisplayableImage &&
                attachment.state == MessageAttachmentState.CACHED &&
                attachment.id !in failedPreviews
            ) {
                InlineImageAttachment(
                    attachment,
                    onClick = {
                        onOpen(attachment.id)
                        preview = attachment
                    },
                    onError = { failedPreviews = failedPreviews + attachment.id },
                )
                ImageAttachmentFooter(
                    attachment = attachment,
                    onShare = {
                        onOpen(attachment.id)
                        shareCachedAttachment(context, attachment)
                    },
                )
            } else {
                FileAttachmentRow(
                    attachment = attachment,
                    onRetry = onRetry,
                    onOpen = {
                        onOpen(attachment.id)
                        openCachedAttachment(context, attachment)
                    },
                    onShare = {
                        onOpen(attachment.id)
                        shareCachedAttachment(context, attachment)
                    },
                )
            }
        }
    }

    preview?.let { attachment ->
        FullscreenImageAttachment(attachment, onDismiss = { preview = null })
    }
}

@Composable
private fun InlineImageAttachment(
    attachment: MessageAttachmentUi,
    onClick: () -> Unit,
    onError: () -> Unit,
) {
    val interactionSource = remember { MutableInteractionSource() }
    val shape = RoundedCornerShape(12.dp)
    val outline = if (isSystemInDarkTheme()) Color.White.copy(alpha = 0.10f) else Color.Black.copy(alpha = 0.10f)
    Surface(
        onClick = onClick,
        interactionSource = interactionSource,
        color = MaterialTheme.colorScheme.surfaceContainerLow,
        shape = shape,
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 148.dp, max = 360.dp)
            .pressScale(interactionSource)
            .testTag("message-image-${attachment.id}"),
    ) {
        AsyncImage(
            model = ImageRequest.Builder(LocalContext.current)
                .data(File(attachment.cachePath))
                .crossfade(true)
                .build(),
            contentDescription = "查看图片 ${attachment.filename}",
            contentScale = ContentScale.Fit,
            onError = { onError() },
            modifier = Modifier
                .fillMaxWidth()
                .clip(shape)
                .border(1.dp, outline, shape),
        )
    }
}

@Composable
private fun FileAttachmentRow(
    attachment: MessageAttachmentUi,
    onRetry: (String) -> Unit,
    onOpen: () -> Unit,
    onShare: () -> Unit,
) {
    val progress = attachment.progress
    val animatedProgress by animateFloatAsState(
        targetValue = progress,
        animationSpec = tween(160),
        label = "download progress",
    )
    val stateLabel = attachment.stateLabel(animatedProgress)
    val rowModifier = Modifier
        .fillMaxWidth()
        .heightIn(min = 64.dp)
        .testTag("message-attachment-${attachment.id}")
        .semantics {
            contentDescription = "文件 ${attachment.filename}"
            stateDescription = stateLabel
            if (attachment.state == MessageAttachmentState.DOWNLOADING) {
                progressBarRangeInfo = ProgressBarRangeInfo(animatedProgress, 0f..1f)
            }
        }

    val content: @Composable () -> Unit = {
        Row(
            modifier = Modifier.padding(start = 4.dp, top = 8.dp, bottom = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            AttachmentStateIcon(attachment.state)
            Column(
                modifier = Modifier
                    .weight(1f)
                    .padding(start = 12.dp),
                verticalArrangement = Arrangement.spacedBy(2.dp),
            ) {
                Text(
                    text = attachment.filename,
                    style = MaterialTheme.typography.labelLarge,
                    color = MaterialTheme.colorScheme.onSurface,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Text(
                    text = "${attachment.typeLabel} · ${formatFileSize(attachment.sizeBytes)} · $stateLabel",
                    style = MaterialTheme.typography.labelMedium.merge(
                        TextStyle(fontFeatureSettings = "tnum"),
                    ),
                    color = if (attachment.state == MessageAttachmentState.FAILED) {
                        MaterialTheme.colorScheme.error
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant
                    },
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                if (attachment.state == MessageAttachmentState.DOWNLOADING) {
                    LinearProgressIndicator(
                        progress = { animatedProgress },
                        color = MaterialTheme.colorScheme.primary,
                        trackColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.16f),
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 5.dp),
                    )
                }
            }
            when (attachment.state) {
                MessageAttachmentState.FAILED,
                MessageAttachmentState.EVICTED,
                -> AttachmentActionButton(
                    onClick = { onRetry(attachment.id) },
                    contentDescription = "重试下载 ${attachment.filename}",
                ) {
                    Icon(Icons.Rounded.Refresh, contentDescription = null)
                }
                MessageAttachmentState.CACHED -> Row {
                    AttachmentActionButton(
                        onClick = onOpen,
                        contentDescription = "打开文件 ${attachment.filename}",
                    ) {
                        Icon(Icons.AutoMirrored.Rounded.OpenInNew, contentDescription = null)
                    }
                    AttachmentActionButton(
                        onClick = onShare,
                        contentDescription = "分享文件 ${attachment.filename}",
                    ) {
                        Icon(Icons.Rounded.Share, contentDescription = null)
                    }
                }
                MessageAttachmentState.PENDING,
                MessageAttachmentState.DOWNLOADING,
                -> Spacer(Modifier.width(48.dp))
            }
        }
    }

    Box(modifier = rowModifier, contentAlignment = Alignment.CenterStart) { content() }
}

@Composable
private fun ImageAttachmentFooter(
    attachment: MessageAttachmentUi,
    onShare: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 48.dp)
            .padding(start = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = attachment.filename,
            style = MaterialTheme.typography.labelLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier.weight(1f),
        )
        AttachmentActionButton(
            onClick = onShare,
            contentDescription = "分享图片 ${attachment.filename}",
        ) {
            Icon(Icons.Rounded.Share, contentDescription = null)
        }
    }
}

@Composable
private fun AttachmentStateIcon(state: MessageAttachmentState) {
    val icon = when (state) {
        MessageAttachmentState.PENDING -> Icons.Rounded.Download
        MessageAttachmentState.DOWNLOADING -> Icons.Rounded.Download
        MessageAttachmentState.CACHED -> Icons.Rounded.Description
        MessageAttachmentState.FAILED -> Icons.Rounded.ErrorOutline
        MessageAttachmentState.EVICTED -> Icons.Rounded.Refresh
    }
    Icon(
        imageVector = icon,
        contentDescription = null,
        tint = when (state) {
            MessageAttachmentState.FAILED -> MaterialTheme.colorScheme.error
            MessageAttachmentState.CACHED -> MaterialTheme.colorScheme.onSurfaceVariant
            MessageAttachmentState.PENDING,
            MessageAttachmentState.DOWNLOADING,
            MessageAttachmentState.EVICTED,
            -> MaterialTheme.colorScheme.primary
        },
        modifier = Modifier.size(24.dp),
    )
}

@Composable
private fun AttachmentActionButton(
    onClick: () -> Unit,
    contentDescription: String,
    content: @Composable () -> Unit,
) {
    val interactionSource = remember { MutableInteractionSource() }
    IconButton(
        onClick = onClick,
        interactionSource = interactionSource,
        modifier = Modifier
            .size(48.dp)
            .pressScale(interactionSource),
    ) {
        Box(Modifier.semantics { this.contentDescription = contentDescription }) { content() }
    }
}

@Composable
private fun FullscreenImageAttachment(
    attachment: MessageAttachmentUi,
    onDismiss: () -> Unit,
) {
    var scale by remember(attachment.id) { mutableFloatStateOf(1f) }
    var offsetX by remember(attachment.id) { mutableFloatStateOf(0f) }
    var offsetY by remember(attachment.id) { mutableFloatStateOf(0f) }
    var loadFailed by remember(attachment.id) { mutableStateOf(false) }
    var viewport by remember(attachment.id) { mutableStateOf(IntSize.Zero) }
    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false),
    ) {
        Surface(
            color = MaterialTheme.colorScheme.surface,
            modifier = Modifier.fillMaxSize(),
        ) {
            Box(Modifier.fillMaxSize()) {
                AsyncImage(
                    model = File(attachment.cachePath),
                    contentDescription = attachment.filename,
                    contentScale = ContentScale.Fit,
                    onError = { loadFailed = true },
                    modifier = Modifier
                        .fillMaxSize()
                        .onSizeChanged { viewport = it }
                        .graphicsLayer {
                            scaleX = scale
                            scaleY = scale
                            translationX = offsetX
                            translationY = offsetY
                        }
                        .pointerInput(attachment.id) {
                            detectTransformGestures { _, pan, zoom, _ ->
                                scale = (scale * zoom).coerceIn(1f, 5f)
                                if (scale == 1f) {
                                    offsetX = 0f
                                    offsetY = 0f
                                } else {
                                    val maxX = viewport.width * (scale - 1f) / 2f
                                    val maxY = viewport.height * (scale - 1f) / 2f
                                    offsetX = (offsetX + pan.x).coerceIn(-maxX, maxX)
                                    offsetY = (offsetY + pan.y).coerceIn(-maxY, maxY)
                                }
                            }
                        }
                        .pointerInput(attachment.id) {
                            detectTapGestures(
                                onDoubleTap = {
                                    scale = if (scale > 1f) 1f else 2.5f
                                    if (scale == 1f) {
                                        offsetX = 0f
                                        offsetY = 0f
                                    }
                                },
                            )
                        },
                )
                if (loadFailed) {
                    Column(
                        modifier = Modifier.align(Alignment.Center),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Icon(
                            Icons.Rounded.ErrorOutline,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.error,
                            modifier = Modifier.size(32.dp),
                        )
                        Text(
                            text = "图片无法显示",
                            style = MaterialTheme.typography.bodyLarge,
                            color = MaterialTheme.colorScheme.onSurface,
                            modifier = Modifier.testTag("fullscreen-image-error"),
                        )
                    }
                }
                val interactionSource = remember { MutableInteractionSource() }
                IconButton(
                    onClick = onDismiss,
                    interactionSource = interactionSource,
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .windowInsetsPadding(WindowInsets.safeDrawing)
                        .padding(8.dp)
                        .size(48.dp)
                        .background(
                            MaterialTheme.colorScheme.surfaceContainerHigh.copy(alpha = 0.92f),
                            RoundedCornerShape(24.dp),
                        )
                        .pressScale(interactionSource),
                ) {
                    Icon(Icons.Rounded.Close, contentDescription = "关闭图片预览")
                }
            }
        }
    }
}

internal val MessageAttachmentUi.isDisplayableImage: Boolean
    get() = contentType.startsWith("image/")

internal val MessageAttachmentUi.progress: Float
    get() = (transferredBytes.toDouble() / sizeBytes.toDouble()).coerceIn(0.0, 1.0).toFloat()

internal val MessageAttachmentUi.typeLabel: String
    get() = when {
        contentType == "image/gif" -> "GIF 图像"
        contentType.startsWith("image/") -> "图像"
        contentType == "application/pdf" -> "PDF"
        contentType.startsWith("video/") -> "视频"
        contentType.startsWith("audio/") -> "音频"
        else -> "文件"
    }

internal fun MessageAttachmentUi.stateLabel(progress: Float = this.progress): String = when (state) {
    MessageAttachmentState.PENDING -> "等待下载"
    MessageAttachmentState.DOWNLOADING -> "下载中 ${(progress * 100).roundToInt()}%"
    MessageAttachmentState.CACHED -> "已下载"
    MessageAttachmentState.FAILED -> "下载失败"
    MessageAttachmentState.EVICTED -> "已清理，需重新下载"
}

private fun openCachedAttachment(context: Context, attachment: MessageAttachmentUi) {
    val uri = FileProvider.getUriForFile(
        context,
        "${BuildConfig.APPLICATION_ID}.files",
        File(attachment.cachePath),
    )
    val intent = Intent(Intent.ACTION_VIEW).apply {
        setDataAndType(uri, attachment.contentType)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    try {
        context.startActivity(intent)
    } catch (_: ActivityNotFoundException) {
        Toast.makeText(context, "没有可打开此文件的应用", Toast.LENGTH_SHORT).show()
    }
}

private fun shareCachedAttachment(context: Context, attachment: MessageAttachmentUi) {
    try {
        context.startActivity(buildShareIntent(context, attachment))
    } catch (_: ActivityNotFoundException) {
        Toast.makeText(context, "没有可分享此文件的应用", Toast.LENGTH_SHORT).show()
    }
}

internal fun buildShareIntent(context: Context, attachment: MessageAttachmentUi): Intent {
    require(attachment.state == MessageAttachmentState.CACHED) { "只有已缓存附件可以分享" }
    val uri = FileProvider.getUriForFile(
        context,
        "${BuildConfig.APPLICATION_ID}.files",
        File(attachment.cachePath),
    )
    val send = Intent(Intent.ACTION_SEND).apply {
        type = attachment.contentType
        putExtra(Intent.EXTRA_STREAM, uri)
        clipData = ClipData.newUri(context.contentResolver, attachment.filename, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    return Intent.createChooser(send, "分享 ${attachment.filename}")
}

private fun formatFileSize(bytes: Long): String = when {
    bytes >= 1024 * 1024 -> "%.1f MB".format(bytes / (1024.0 * 1024.0))
    bytes >= 1024 -> "%.1f KB".format(bytes / 1024.0)
    else -> "$bytes B"
}
