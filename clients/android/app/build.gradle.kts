plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.ksp)
}

val releaseKeystorePath = providers.environmentVariable("AKASHIC_ANDROID_KEYSTORE_PATH").orNull
val releaseStorePassword = providers.environmentVariable("AKASHIC_ANDROID_STORE_PASSWORD").orNull
val releaseKeyAlias = providers.environmentVariable("AKASHIC_ANDROID_KEY_ALIAS").orNull
val releaseKeyPassword = providers.environmentVariable("AKASHIC_ANDROID_KEY_PASSWORD").orNull
val hasReleaseSigning = listOf(
    releaseKeystorePath,
    releaseStorePassword,
    releaseKeyAlias,
    releaseKeyPassword,
).all { !it.isNullOrBlank() }
if (!hasReleaseSigning && gradle.startParameter.taskNames.any { it.contains("release", ignoreCase = true) }) {
    throw GradleException("Release signing environment is required")
}

android {
    namespace = "com.akashic.mobile"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.akashic.mobile"
        minSdk = 26
        targetSdk = 36
        versionCode = 8
        versionName = "0.6.2"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    signingConfigs {
        if (hasReleaseSigning) {
            create("release") {
                storeFile = file(requireNotNull(releaseKeystorePath))
                storePassword = releaseStorePassword
                keyAlias = releaseKeyAlias
                keyPassword = releaseKeyPassword
            }
        }
    }

    buildTypes {
        debug {
            buildConfigField("boolean", "ALLOW_INSECURE_WS", "true")
        }
        release {
            isMinifyEnabled = true
            signingConfig = signingConfigs.findByName("release")
            buildConfigField("boolean", "ALLOW_INSECURE_WS", "false")
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        buildConfig = true
        compose = true
    }

    packaging {
        resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
    }

    sourceSets {
        getByName("androidTest").assets.srcDir("$projectDir/schemas")
    }
}

kotlin {
    jvmToolchain(17)
}

ksp {
    arg("room.schemaLocation", "$projectDir/schemas")
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)

    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.material.icons.extended)
    debugImplementation(libs.androidx.compose.ui.tooling)
    debugImplementation(libs.androidx.compose.ui.test.manifest)

    implementation(libs.androidx.room.runtime)
    implementation(libs.androidx.room.ktx)
    ksp(libs.androidx.room.compiler)
    implementation(libs.androidx.datastore.preferences)

    implementation(libs.kotlinx.serialization.json)
    implementation(libs.kotlinx.coroutines.android)
    implementation(platform(libs.okhttp.bom))
    implementation(libs.okhttp)
    implementation(libs.google.material)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)
    implementation(libs.zxing.core)
    implementation(libs.markdown.renderer.m3)
    implementation(libs.latex.base)
    implementation(libs.latex.parser)
    implementation(libs.latex.renderer)
    implementation(libs.coil.compose)
    implementation(libs.coil.gif)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.room.testing)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
}
