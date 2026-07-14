package com.akashic.mobile.ui.design

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.unit.dp

private val LightColors = lightColorScheme(
    primary = AkashicColors.Primary,
    onPrimary = AkashicColors.OnPrimary,
    primaryContainer = AkashicColors.PrimaryContainer,
    onPrimaryContainer = AkashicColors.OnPrimaryContainer,
    tertiary = AkashicColors.Tertiary,
    onTertiary = AkashicColors.OnTertiary,
    tertiaryContainer = AkashicColors.TertiaryContainer,
    onTertiaryContainer = AkashicColors.OnTertiaryContainer,
    surface = AkashicColors.Surface,
    onSurface = AkashicColors.OnSurface,
    surfaceVariant = AkashicColors.SurfaceVariant,
    onSurfaceVariant = AkashicColors.OnSurfaceVariant,
    surfaceContainerLowest = AkashicColors.Surface,
    surfaceContainerLow = AkashicColors.SurfaceVariant.copy(alpha = 0.42f),
    surfaceContainer = AkashicColors.SurfaceVariant.copy(alpha = 0.62f),
    surfaceContainerHigh = AkashicColors.SurfaceVariant.copy(alpha = 0.82f),
    surfaceContainerHighest = AkashicColors.SurfaceVariant,
    outline = AkashicColors.Outline,
    error = AkashicColors.Error,
    onError = AkashicColors.OnError,
    errorContainer = AkashicColors.ErrorContainer,
    onErrorContainer = AkashicColors.OnErrorContainer,
)

private val DarkColors = darkColorScheme(
    primary = AkashicColors.DarkPrimary,
    onPrimary = AkashicColors.DarkOnPrimary,
    primaryContainer = AkashicColors.DarkPrimaryContainer,
    onPrimaryContainer = AkashicColors.DarkOnPrimaryContainer,
    tertiary = AkashicColors.DarkTertiary,
    onTertiary = AkashicColors.DarkOnTertiary,
    tertiaryContainer = AkashicColors.DarkTertiaryContainer,
    onTertiaryContainer = AkashicColors.DarkOnTertiaryContainer,
    surface = AkashicColors.DarkSurface,
    onSurface = AkashicColors.DarkOnSurface,
    surfaceVariant = AkashicColors.DarkSurfaceVariant,
    onSurfaceVariant = AkashicColors.DarkOnSurfaceVariant,
    surfaceContainerLowest = AkashicColors.DarkSurface,
    surfaceContainerLow = AkashicColors.DarkSurfaceVariant.copy(alpha = 0.52f),
    surfaceContainer = AkashicColors.DarkSurfaceVariant.copy(alpha = 0.72f),
    surfaceContainerHigh = AkashicColors.DarkSurfaceVariant.copy(alpha = 0.88f),
    surfaceContainerHighest = AkashicColors.DarkSurfaceVariant,
    outline = AkashicColors.DarkOutline,
    error = AkashicColors.DarkError,
    onError = AkashicColors.DarkOnError,
    errorContainer = AkashicColors.DarkErrorContainer,
    onErrorContainer = AkashicColors.DarkOnErrorContainer,
)

private val AkashicShapes = Shapes(
    extraSmall = RoundedCornerShape(4.dp),
    small = RoundedCornerShape(8.dp),
    medium = RoundedCornerShape(12.dp),
    large = RoundedCornerShape(20.dp),
    extraLarge = RoundedCornerShape(28.dp),
)

@Composable
fun AkashicTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    MaterialTheme(
        colorScheme = if (darkTheme) DarkColors else LightColors,
        shapes = AkashicShapes,
        content = content,
    )
}
