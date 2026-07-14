package com.akashic.mobile.ui.design

import androidx.compose.ui.graphics.Color

// 由规格中的 OKLCH token 转换并校验为 sRGB ARGB；Compose 运行时不做色域转换。
internal object AkashicColors {
    val Primary = Color(0xFF2463AE)
    val OnPrimary = Color(0xFFF7FAFE)
    val PrimaryContainer = Color(0xFFCFE3FE)
    val OnPrimaryContainer = Color(0xFF0C2A4E)
    val Tertiary = Color(0xFF9655C8)
    val OnTertiary = Color(0xFFFCF9FD)
    val TertiaryContainer = Color(0xFFE9D6F2)
    val OnTertiaryContainer = Color(0xFF35203E)
    val Surface = Color(0xFFF7FAFE)
    val OnSurface = Color(0xFF161A1F)
    val SurfaceVariant = Color(0xFFE1E9F3)
    val OnSurfaceVariant = Color(0xFF313A46)
    val Outline = Color(0xFF6A727D)
    val Error = Color(0xFFA12F2F)
    val OnError = Color(0xFFFEF9F8)
    val ErrorContainer = Color(0xFFFBD3CF)
    val OnErrorContainer = Color(0xFF4A1615)

    val DarkPrimary = Color(0xFF97C1F7)
    val DarkOnPrimary = Color(0xFF061F3D)
    val DarkPrimaryContainer = Color(0xFF183860)
    val DarkOnPrimaryContainer = Color(0xFFCFE3FE)
    val DarkTertiary = Color(0xFFD8B4FE)
    val DarkOnTertiary = Color(0xFF291631)
    val DarkTertiaryContainer = Color(0xFF452C51)
    val DarkOnTertiaryContainer = Color(0xFFE9D6F2)
    val DarkSurface = Color(0xFF0B0D11)
    val DarkOnSurface = Color(0xFFE3E8EF)
    val DarkSurfaceVariant = Color(0xFF1B2026)
    val DarkOnSurfaceVariant = Color(0xFFB3BFCE)
    val DarkOutline = Color(0xFF737B86)
    val DarkError = Color(0xFFEDAAA4)
    val DarkOnError = Color(0xFF371210)
    val DarkErrorContainer = Color(0xFF5A2522)
    val DarkOnErrorContainer = Color(0xFFFBD3CF)
}
