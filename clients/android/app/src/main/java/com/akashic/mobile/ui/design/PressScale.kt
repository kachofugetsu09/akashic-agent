package com.akashic.mobile.ui.design

import android.animation.ValueAnimator
import androidx.compose.animation.core.CubicBezierEasing
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.snap
import androidx.compose.animation.core.tween
import androidx.compose.foundation.interaction.InteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.graphicsLayer

@Composable
fun Modifier.pressScale(
    interactionSource: InteractionSource,
    enabled: Boolean = true,
): Modifier {
    val pressed by interactionSource.collectIsPressedAsState()
    val animationsEnabled = remember { ValueAnimator.areAnimatorsEnabled() }
    val scale by animateFloatAsState(
        targetValue = if (enabled && pressed) 0.96f else 1f,
        animationSpec = if (animationsEnabled) {
            tween(durationMillis = 150, easing = CubicBezierEasing(0.2f, 0f, 0f, 1f))
        } else {
            snap()
        },
        label = "press scale",
    )
    return graphicsLayer {
        scaleX = scale
        scaleY = scale
    }
}
