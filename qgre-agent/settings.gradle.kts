// qgre-agent — a pure LLM agent (aisdk-kotlin) that drives the qgre data-pipeline rebuild.
// Standalone JVM Gradle project (sibling to the Rust festival-factory/). Depends on the
// aisdk-kotlin port published to mavenLocal (./gradlew publishToMavenLocal in the SDK repo).
rootProject.name = "qgre-agent"

pluginManagement {
    repositories {
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        mavenLocal()   // ai.torad:aisdk-kotlin-jvm:0.1.0-SNAPSHOT lives here
        mavenCentral()
    }
}
