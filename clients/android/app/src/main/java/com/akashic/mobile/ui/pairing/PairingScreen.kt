package com.akashic.mobile.ui.pairing

import android.Manifest
import android.content.ClipboardManager
import android.content.pm.PackageManager
import android.util.Log
import android.util.Size
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.QrCodeScanner
import androidx.compose.material3.Button
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.google.zxing.BinaryBitmap
import com.google.zxing.BarcodeFormat
import com.google.zxing.ChecksumException
import com.google.zxing.DecodeHintType
import com.google.zxing.FormatException
import com.google.zxing.MultiFormatReader
import com.google.zxing.NotFoundException
import com.google.zxing.PlanarYUVLuminanceSource
import com.google.zxing.common.HybridBinarizer
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

@Composable
fun PairingScreen(
    scanGeneration: Long,
    confirmationCode: String?,
    errorMessage: String?,
    onQrCode: (String) -> Unit,
) {
    var manualInput by remember { mutableStateOf(false) }
    if (confirmationCode != null) {
        PairingConfirmation(confirmationCode, errorMessage)
    } else if (manualInput) {
        PairingManualInput(
            errorMessage = errorMessage,
            onBack = { manualInput = false },
            onQrCode = onQrCode,
        )
    } else {
        CameraPermissionGate(scanGeneration, errorMessage, onQrCode) { manualInput = true }
    }
}

@Composable
private fun CameraPermissionGate(
    scanGeneration: Long,
    errorMessage: String?,
    onQrCode: (String) -> Unit,
    onManualInput: () -> Unit,
) {
    val context = LocalContext.current
    var granted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val request = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    LaunchedEffect(Unit) { if (!granted) request.launch(Manifest.permission.CAMERA) }
    if (granted) {
        CameraQrScanner(scanGeneration, errorMessage, onQrCode, onManualInput)
    } else {
        PairingExplanation(
            errorMessage = errorMessage,
            requestPermission = { request.launch(Manifest.permission.CAMERA) },
            onManualInput = onManualInput,
        )
    }
}

@Composable
private fun CameraQrScanner(
    scanGeneration: Long,
    errorMessage: String?,
    onQrCode: (String) -> Unit,
    onManualInput: () -> Unit,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val executor = remember(scanGeneration) { Executors.newSingleThreadExecutor() }
    val accepted = remember(scanGeneration) { AtomicBoolean(false) }
    val frameLogged = remember(scanGeneration) { AtomicBoolean(false) }
    val previewView = remember { PreviewView(context) }

    DisposableEffect(lifecycleOwner, previewView, scanGeneration) {
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener(
            {
                val provider = providerFuture.get()
                val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(Size(1280, 720))
                    .setOutputImageRotationEnabled(true)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                val reader = MultiFormatReader().apply {
                    setHints(
                        mapOf(
                            DecodeHintType.POSSIBLE_FORMATS to listOf(BarcodeFormat.QR_CODE),
                            DecodeHintType.TRY_HARDER to true,
                        ),
                    )
                }
                analysis.setAnalyzer(executor) analyzer@{ image ->
                    if (accepted.get()) {
                        image.close()
                        return@analyzer
                    }
                    try {
                        val plane = image.planes[0]
                        if (frameLogged.compareAndSet(false, true)) {
                            Log.i(
                                PAIRING_LOG_TAG,
                                "相机首帧: ${image.width}x${image.height}, rotation=${image.imageInfo.rotationDegrees}, rowStride=${plane.rowStride}",
                            )
                        }
                        val bytes = ByteArray(plane.buffer.remaining())
                        plane.buffer.get(bytes)
                        val source = PlanarYUVLuminanceSource(
                            bytes,
                            plane.rowStride,
                            image.height,
                            0,
                            0,
                            image.width,
                            image.height,
                            false,
                        )
                        val candidate = reader.decodeWithState(BinaryBitmap(HybridBinarizer(source))).text.trim()
                        Log.i(
                            PAIRING_LOG_TAG,
                            "二维码已解码: length=${candidate.length}, object=${isJsonObjectCandidate(candidate)}",
                        )
                        if (isJsonObjectCandidate(candidate)) {
                            Log.i(PAIRING_LOG_TAG, "接受配对二维码: length=${candidate.length}")
                            if (accepted.compareAndSet(false, true)) onQrCode(candidate)
                        } else {
                            reader.reset()
                        }
                    } catch (_: NotFoundException) {
                        reader.reset()
                    } catch (_: ChecksumException) {
                        reader.reset()
                    } catch (_: FormatException) {
                        reader.reset()
                    } finally {
                        image.close()
                    }
                }
                provider.unbindAll()
                provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            },
            ContextCompat.getMainExecutor(context),
        )
        onDispose {
            if (providerFuture.isDone) providerFuture.get().unbindAll()
            executor.shutdownNow()
        }
    }

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
        Canvas(Modifier.align(Alignment.Center).size(260.dp)) {
            drawRoundRect(
                color = Color.White.copy(alpha = 0.9f),
                style = androidx.compose.ui.graphics.drawscope.Stroke(width = 3.dp.toPx()),
                cornerRadius = CornerRadius(24.dp.toPx()),
            )
        }
        Column(
            modifier = Modifier.align(Alignment.BottomCenter).padding(horizontal = 28.dp, vertical = 48.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text("扫描电脑 WebChat 中的二维码", color = Color.White, style = MaterialTheme.typography.titleMedium)
            Text(
                errorMessage ?: "二维码只使用一次；配对成功后不会保存其中的 secret。",
                color = Color.White.copy(alpha = 0.78f),
                textAlign = TextAlign.Center,
                style = MaterialTheme.typography.bodyMedium,
            )
            TextButton(onClick = onManualInput) {
                Text("手动输入配对内容", color = Color.White)
            }
        }
    }
}

private const val PAIRING_LOG_TAG = "AkashicPairing"

private fun isJsonObjectCandidate(value: String): Boolean {
    val trimmed = value.trim()
    return trimmed.startsWith("{") && trimmed.endsWith("}")
}

@Composable
private fun PairingExplanation(
    errorMessage: String?,
    requestPermission: () -> Unit,
    onManualInput: () -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxSize().padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Icon(Icons.Rounded.QrCodeScanner, contentDescription = null, modifier = Modifier.size(48.dp))
        Text("连接你的电脑", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold)
        Text(
            errorMessage ?: "允许相机权限后，扫描 WebChat 显示的一次性二维码。",
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(vertical = 16.dp),
        )
        Button(onClick = requestPermission) { Text("允许相机并扫码") }
        TextButton(onClick = onManualInput) { Text("手动输入配对内容") }
    }
}

@Composable
private fun PairingManualInput(
    errorMessage: String?,
    onBack: () -> Unit,
    onQrCode: (String) -> Unit,
) {
    val context = LocalContext.current
    val clipboard = remember(context) { context.getSystemService(ClipboardManager::class.java) }
    var value by remember { mutableStateOf("") }
    var inputError by remember { mutableStateOf<String?>(null) }
    val updateValue = { candidate: String ->
        if (candidate.toByteArray(Charsets.UTF_8).size <= MAX_MANUAL_PAIRING_BYTES) {
            value = candidate
            inputError = null
        } else {
            inputError = "配对内容不能超过 32 KB"
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .imePadding()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 28.dp, vertical = 32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Text("手动输入配对内容", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold)
        Text(
            "把扫描器复制出的完整 JSON 粘贴到这里；仍会执行相同的签名、有效期和证书校验。",
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 12.dp, bottom = 18.dp),
        )
        OutlinedTextField(
            value = value,
            onValueChange = updateValue,
            modifier = Modifier.fillMaxWidth(),
            minLines = 6,
            maxLines = 12,
            label = { Text("配对 JSON") },
            isError = inputError != null,
        )
        val message = inputError ?: errorMessage
        if (message != null) {
            Text(
                message,
                color = MaterialTheme.colorScheme.error,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(top = 8.dp),
            )
        }
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(4.dp),
            modifier = Modifier.padding(top = 14.dp),
        ) {
            FilledTonalButton(onClick = {
                val pasted = clipboard?.primaryClip?.getItemAt(0)?.coerceToText(context)?.toString().orEmpty()
                if (pasted.isBlank()) inputError = "剪贴板没有可用的配对内容" else updateValue(pasted)
            }) { Text("从剪贴板粘贴") }
            Button(onClick = {
                val candidate = value.trim()
                if (candidate.isEmpty()) inputError = "请输入或粘贴配对内容" else onQrCode(candidate)
            }) { Text("继续连接") }
            TextButton(onClick = onBack) { Text("返回扫码") }
        }
    }
}

@Composable
private fun PairingConfirmation(code: String, errorMessage: String?) {
    Column(
        modifier = Modifier.fillMaxSize().padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text("在电脑上确认", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold)
        Text(
            code.chunked(3).joinToString(" "),
            style = MaterialTheme.typography.displayMedium,
            color = MaterialTheme.colorScheme.primary,
            modifier = Modifier
                .padding(vertical = 24.dp)
                .background(MaterialTheme.colorScheme.primaryContainer, RoundedCornerShape(24.dp))
                .padding(horizontal = 28.dp, vertical = 18.dp),
        )
        Text(
            errorMessage ?: "手机和 WebChat 显示相同数字时，在电脑上点击确认。",
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

private const val MAX_MANUAL_PAIRING_BYTES = 32 * 1024
