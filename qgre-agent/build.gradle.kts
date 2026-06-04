// Versions pinned to match the aisdk-kotlin port (Kotlin 2.3.21, coroutines/serialization 1.11.0,
// Ktor 3.5.0). The SDK exposes ktor-client-core as `api` but ships NO JVM engine, so we add
// ktor-client-cio for the real HTTP the OpenRouter provider + the serve.py bridge need.
plugins {
    kotlin("jvm") version "2.3.21"
    kotlin("plugin.serialization") version "2.3.21"
    application
    jacoco
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    implementation("ai.torad:aisdk-kotlin-jvm:0.1.0-SNAPSHOT")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.11.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.11.0")
    implementation("io.ktor:ktor-client-cio:3.5.0")            // JVM HTTP engine (SDK ships none)
    runtimeOnly("org.slf4j:slf4j-nop:2.0.16")                  // silence ktor-cio's SLF4J no-provider warning

    testImplementation(kotlin("test"))
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.11.0")
    testImplementation("io.ktor:ktor-client-mock:3.5.0")        // fake the serve.py HTTP boundary in tests
}

application {
    mainClass.set("ai.torad.qgre.MainKt")
}

tasks.test {
    useJUnitPlatform()
    finalizedBy(tasks.jacocoTestReport)
}

tasks.jacocoTestReport {
    dependsOn(tasks.test)
    reports {
        xml.required.set(true)
        html.required.set(true)
    }
}
