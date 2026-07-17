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

val repositoryRoot = rootProject.projectDir.parentFile.parentFile
val generatedMobileWebAssets = layout.buildDirectory.dir("generated/mobileWebAssets")
val buildMobileWeb by tasks.registering(Exec::class) {
    workingDir = repositoryRoot
    environment("AKASHIC_MOBILE_WEB_OUT_DIR", generatedMobileWebAssets.get().asFile.absolutePath)
    commandLine("npm", "run", "build:mobile-web")
    inputs.file(repositoryRoot.resolve("package.json"))
    inputs.file(repositoryRoot.resolve("package-lock.json"))
    inputs.dir(repositoryRoot.resolve("frontend/chat/src"))
    inputs.file(repositoryRoot.resolve("frontend/chat/mobile.html"))
    inputs.file(repositoryRoot.resolve("frontend/chat/vite.mobile.config.ts"))
    inputs.file(repositoryRoot.resolve("frontend/chat/postcss.config.cjs"))
    inputs.file(repositoryRoot.resolve("frontend/chat/tailwind.config.cjs"))
    inputs.file(repositoryRoot.resolve("tsconfig.json"))
    outputs.dir(generatedMobileWebAssets)
}
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
        versionCode = 23
        versionName = "0.8.2"

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
            applicationIdSuffix = ".debug"
            versionNameSuffix = "-debug"
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
        getByName("main").assets.srcDir(generatedMobileWebAssets)
    }
}

tasks.named("preBuild").configure {
    dependsOn(buildMobileWeb)
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
    implementation(libs.androidx.webkit)

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
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.room.testing)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
}
