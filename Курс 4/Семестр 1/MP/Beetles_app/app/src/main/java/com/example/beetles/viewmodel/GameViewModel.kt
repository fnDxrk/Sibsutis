package com.example.beetles.viewmodel

import android.app.Application
import androidx.compose.runtime.mutableStateListOf
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.beetles.data.Beetle
import com.example.beetles.data.GameState
import com.example.beetles.data.SettingsData
import com.example.beetles.repository.SettingsRepository
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlin.math.atan2
import kotlin.random.Random

class GameViewModel(application: Application) : AndroidViewModel(application) {
    private val repository = SettingsRepository(application)
    private val _gameState = MutableStateFlow(GameState())
    val gameState: StateFlow<GameState> = _gameState.asStateFlow()
    val beetles = mutableStateListOf<Beetle>()

    private var gameLoopJob: Job? = null
    private var timerJob: Job? = null
    private var spawnJob: Job? = null
    private var countdownJob: Job? = null
    private var nextBeetleId = 0
    private var screenWidth = 0f
    private var screenHeight = 0f
    private val beetleSize = 80f
    private val topBarHeight = 80f
    private val bottomBarHeight = 80f
    private var settingsCache: SettingsData? = null

    private var spawnDelayMs = 1000L
    private var directionChangeInterval = 2f
    private var wallDamage = 0.3f
    private var isInitialized = false

    fun initGame(width: Float, height: Float) {
        if (isInitialized) return

        isInitialized = true
        screenWidth = width
        screenHeight = height

        viewModelScope.launch {
            val settings = settingsCache ?: repository.settingsFlow.first().also {
                settingsCache = it
            }

            calculateDifficultyParameters(settings.gameSpeed)

            _gameState.value = GameState(
                maxBeetles = settings.maxBeetles,
                gameSpeed = settings.gameSpeed,
                timeLeft = settings.roundDuration,
                isGameStarted = false,
                countdown = 3
            )

            startGame()
        }
    }

    private fun calculateDifficultyParameters(gameSpeed: Float) {
        when {
            gameSpeed <= 3f -> {
                spawnDelayMs = 2500L
                directionChangeInterval = 1.2f
                wallDamage = 0.5f
            }
            gameSpeed <= 7f -> {
                spawnDelayMs = 1500L
                directionChangeInterval = 2.0f
                wallDamage = 0.3f
            }
            else -> {
                spawnDelayMs = 800L
                directionChangeInterval = 3.5f
                wallDamage = 0.15f
            }
        }
    }

    private fun startGame() {
        stopAllJobs()

        countdownJob = viewModelScope.launch {
            for (i in 3 downTo 1) {
                _gameState.value = _gameState.value.copy(countdown = i)
                delay(1000)
            }
            _gameState.value = _gameState.value.copy(isGameStarted = true, countdown = 0)
            startGameLoop()
        }
    }

    private fun startGameLoop() {
        gameLoopJob = viewModelScope.launch {
            while (!_gameState.value.isGameOver && _gameState.value.isGameStarted) {
                updateGame()
                delay(16)
            }
        }

        timerJob = viewModelScope.launch {
            while (!_gameState.value.isGameOver && _gameState.value.isGameStarted) {
                delay(1000)
                val state = _gameState.value
                val newTime = state.timeLeft - 1
                _gameState.value = state.copy(timeLeft = newTime)

                if (newTime <= 0) {
                    endGame()
                }
            }
        }

        spawnJob = viewModelScope.launch {
            while (!_gameState.value.isGameOver && _gameState.value.isGameStarted) {
                val state = _gameState.value
                if (beetles.count { it.isAlive } < state.maxBeetles) {
                    spawnBeetle()
                    delay(spawnDelayMs)
                } else {
                    delay(200)
                }
            }
        }
    }

    private fun updateGame() {
        val state = _gameState.value

        beetles.forEach { beetle ->
            if (!beetle.isAlive) return@forEach

            beetle.lifeTime -= 0.016f
            if (beetle.lifeTime <= 0) {
                beetle.isAlive = false
                _gameState.value = state.copy(score = state.score - 5)
                return@forEach
            }

            beetle.directionChangeTimer -= 0.016f
            if (beetle.directionChangeTimer <= 0 &&
                beetle.x > 20 && beetle.x < screenWidth - beetleSize - 20 &&
                beetle.y > topBarHeight + 20 && beetle.y < screenHeight - bottomBarHeight - beetleSize - 20) {

                val angle = Random.nextFloat() * 360f
                val speed = state.gameSpeed * 2.5f
                beetle.speedX = kotlin.math.cos(Math.toRadians(angle.toDouble())).toFloat() * speed
                beetle.speedY = kotlin.math.sin(Math.toRadians(angle.toDouble())).toFloat() * speed
                beetle.directionChangeTimer = Random.nextFloat() * 2f + directionChangeInterval
                beetle.rotation = angle + 90f
            }

            val nextX = beetle.x + beetle.speedX
            val nextY = beetle.y + beetle.speedY
            var bounced = false

            if (nextX < 0 || nextX > screenWidth - beetleSize) {
                beetle.speedX = -beetle.speedX
                beetle.x = beetle.x.coerceIn(0f, screenWidth - beetleSize)
                beetle.rotation = Math.toDegrees(
                    atan2(beetle.speedY.toDouble(), beetle.speedX.toDouble())
                ).toFloat() + 90f
                bounced = true
            } else {
                beetle.x = nextX
            }

            if (nextY < topBarHeight || nextY > screenHeight - bottomBarHeight - beetleSize) {
                beetle.speedY = -beetle.speedY
                beetle.y = beetle.y.coerceIn(topBarHeight, screenHeight - bottomBarHeight - beetleSize)
                beetle.rotation = Math.toDegrees(
                    atan2(beetle.speedY.toDouble(), beetle.speedX.toDouble())
                ).toFloat() + 90f
                bounced = true
            } else {
                beetle.y = nextY
            }

            if (bounced) {
                beetle.lifeTime -= wallDamage
            }
        }

        beetles.removeAll { !it.isAlive }
    }

    private fun spawnBeetle() {
        val state = _gameState.value
        val angle = Random.nextFloat() * 360f
        val speed = state.gameSpeed * 2.5f
        val speedX = kotlin.math.cos(Math.toRadians(angle.toDouble())).toFloat() * speed
        val speedY = kotlin.math.sin(Math.toRadians(angle.toDouble())).toFloat() * speed
        val lifeTime = (10f - state.gameSpeed * 0.6f).coerceIn(4f, 9f)

        val beetle = Beetle(
            id = nextBeetleId++,
            x = Random.nextFloat() * (screenWidth - beetleSize),
            y = Random.nextFloat() * (screenHeight - topBarHeight - bottomBarHeight - beetleSize) + topBarHeight,
            speedX = speedX,
            speedY = speedY,
            rotation = angle + 90f,
            directionChangeTimer = Random.nextFloat() * 2f + directionChangeInterval,
            lifeTime = lifeTime
        )

        beetles.add(beetle)
    }

    fun onBeetleClicked(beetleId: Int) {
        val beetle = beetles.find { it.id == beetleId && it.isAlive }
        if (beetle != null) {
            beetle.isAlive = false
            val state = _gameState.value
            _gameState.value = state.copy(score = state.score + 10)
        }
    }

    fun onMissClick() {
        if (!_gameState.value.isGameStarted) return
        val state = _gameState.value
        _gameState.value = state.copy(score = state.score - 5)
    }

    private fun endGame() {
        _gameState.value = _gameState.value.copy(isGameOver = true)
        stopAllJobs()
    }

    private fun stopAllJobs() {
        gameLoopJob?.cancel()
        timerJob?.cancel()
        spawnJob?.cancel()
        countdownJob?.cancel()
    }

    fun resetGame() {
        stopAllJobs()
        beetles.clear()
        nextBeetleId = 0
        isInitialized = false
        settingsCache = null
        _gameState.value = GameState()
        screenWidth = 0f
        screenHeight = 0f
    }

    override fun onCleared() {
        super.onCleared()
        stopAllJobs()
    }
}
