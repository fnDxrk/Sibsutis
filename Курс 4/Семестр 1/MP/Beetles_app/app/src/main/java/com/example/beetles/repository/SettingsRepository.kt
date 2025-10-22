package com.example.beetles.repository

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.example.beetles.data.SettingsData
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore(name = "game_settings")

class SettingsRepository(private val context: Context) {

    companion object {
        private val GAME_SPEED = floatPreferencesKey("game_speed")
        private val MAX_BEETLES = intPreferencesKey("max_beetles")
        private val BONUS_INTERVAL = intPreferencesKey("bonus_interval")
        private val ROUND_DURATION = intPreferencesKey("round_duration")
    }

    val settingsFlow: Flow<SettingsData> = context.dataStore.data.map { preferences ->
        SettingsData(
            gameSpeed = preferences[GAME_SPEED] ?: 5f,
            maxBeetles = preferences[MAX_BEETLES] ?: 10,
            bonusInterval = preferences[BONUS_INTERVAL] ?: 5,
            roundDuration = preferences[ROUND_DURATION] ?: 60
        )
    }

    suspend fun saveSettings(settings: SettingsData) {
        context.dataStore.edit { preferences ->
            preferences[GAME_SPEED] = settings.gameSpeed
            preferences[MAX_BEETLES] = settings.maxBeetles
            preferences[BONUS_INTERVAL] = settings.bonusInterval
            preferences[ROUND_DURATION] = settings.roundDuration
        }
    }
}