package com.example.beetles.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.beetles.data.SettingsData
import com.example.beetles.repository.SettingsRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class SettingsViewModel(application: Application) : AndroidViewModel(application) {

    private val repository = SettingsRepository(application)

    private val _settings = MutableStateFlow(SettingsData())
    val settings: StateFlow<SettingsData> = _settings.asStateFlow()

    init {
        loadSettings()
    }

    private fun loadSettings() {
        viewModelScope.launch {
            repository.settingsFlow.collect { loadedSettings ->
                _settings.value = loadedSettings
            }
        }
    }

    fun updateGameSpeed(speed: Float) {
        val updated = _settings.value.copy(gameSpeed = speed)
        _settings.value = updated
        saveSettings(updated)
    }

    fun updateMaxBeetles(count: Int) {
        val updated = _settings.value.copy(maxBeetles = count)
        _settings.value = updated
        saveSettings(updated)
    }

    fun updateBonusInterval(interval: Int) {
        val updated = _settings.value.copy(bonusInterval = interval)
        _settings.value = updated
        saveSettings(updated)
    }

    fun updateRoundDuration(duration: Int) {
        val updated = _settings.value.copy(roundDuration = duration)
        _settings.value = updated
        saveSettings(updated)
    }

    private fun saveSettings(settings: SettingsData) {
        viewModelScope.launch {
            repository.saveSettings(settings)
        }
    }
}