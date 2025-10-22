package com.example.beetles.data

data class Beetle(
    val id: Int,
    var x: Float,
    var y: Float,
    var speedX: Float,
    var speedY: Float,
    var rotation: Float = 0f,
    var directionChangeTimer: Float = 0f,
    var lifeTime: Float = 5f,
    var isAlive: Boolean = true
)
