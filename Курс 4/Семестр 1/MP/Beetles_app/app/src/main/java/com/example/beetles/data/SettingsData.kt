package com.example.beetles.data

data class SettingsData(
    val gameSpeed: Float = 5f,
    val maxBeetles: Int = 10,
    val bonusInterval: Int = 5,
    val roundDuration: Int = 60
)
