package com.example.beetles.data

data class GameState(
    val beetles: List<Beetle> = emptyList(),
    val score: Int = 0,
    val timeLeft: Int = 60,
    val isGameOver: Boolean = false,
    val isGameStarted: Boolean = false,
    val countdown: Int = 3,
    val maxBeetles: Int = 10,
    val gameSpeed: Float = 5f
)

