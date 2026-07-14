package com.akashic.mobile.ui.pairing

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.QrCodeScanner
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
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
import com.google.zxing.ChecksumException
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
    if (confirmationCode != null) {
        PairingConfirmation(confirmationCode, errorMessage)
    } else {
        CameraPermissionGate(scanGeneration, errorMessage, onQrCode)
    }
}

@Composable
private fun CameraPermissionGate(scanGeneration: Long, errorMessage: String?, onQrCode: (String) -> Unit) {
    val context = LocalContext.current
    var granted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val request = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    LaunchedEffect(Unit) { if (!granted) request.launch(Manifest.permission.CAMERA) }
    if (granted) {
        CameraQrScanner(scanGeneration, errorMessage, onQrCode)
    } else {
        PairingExplanation(errorMessage) { request.launch(Manifest.permission.CAMERA) }
    }
}

@Composable
private fun CameraQrScanner(scanGeneration: Long, errorMessage: String?, onQrCode: (String) -> Unit) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val executor = remember(scanGeneration) { Executors.newSingleThreadExecutor() }
    val accepted = remember(scanGeneration) { AtomicBoolean(false) }
    val previewView = remember { PreviewView(context) }

    DisposableEffect(lifecycleOwner, previewView, scanGeneration) {
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener(
            {
                val provider = providerFuture.get()
                val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                val reader = MultiFormatReader()
                analysis.setAnalyzer(executor) analyzer@{ image ->
                    if (accepted.get()) {
                        image.close()
                        return@analyzer
                    }
                    try {
                        val plane = image.planes[0]
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
                        val result = reader.decodeWithState(BinaryBitmap(HybridBinarizer(source)))
                        if (accepted.compareAndSet(false, true)) onQrCode(result.text)
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
        }
    }
}

@Composable
private fun PairingExplanation(errorMessage: String?, requestPermission: () -> Unit) {
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
