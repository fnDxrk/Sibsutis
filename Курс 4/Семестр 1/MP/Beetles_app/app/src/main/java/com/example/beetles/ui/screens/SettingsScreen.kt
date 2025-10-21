package com.example.beetles.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp

@Composable
fun SettingsScreen() {
    var gameSpeed by remember { mutableFloatStateOf(5f) }
    var maxBeetles by remember { mutableFloatStateOf(10f) }
    var bonusInterval by remember { mutableFloatStateOf(5f) }
    var roundDuration by remember { mutableFloatStateOf(60f) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text(
            text = "Настройки игры",
            modifier = Modifier.fillMaxWidth(),
            textAlign = TextAlign.Center,
            style = MaterialTheme.typography.headlineMedium
        )

        Card(
            modifier = Modifier.fillMaxWidth(),
            elevation = CardDefaults.cardElevation(2.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "Скорость игры: ${gameSpeed.toInt()}",
                    style = MaterialTheme.typography.titleMedium
                )
                Slider(
                    value = gameSpeed,
                    onValueChange = { gameSpeed = it },
                    valueRange = 1f..10f,
                    steps = 8,
                    modifier = Modifier.fillMaxWidth()
                )
                Text(
                    text = "От 1 (медленно) до 10 (быстро)",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.secondary
                )
            }
        }

        Card(
            modifier = Modifier.fillMaxWidth(),
            elevation = CardDefaults.cardElevation(2.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "Макс. количество тараканов: ${maxBeetles.toInt()}",
                    style = MaterialTheme.typography.titleMedium
                )
                Slider(
                    value = maxBeetles,
                    onValueChange = { maxBeetles = it },
                    valueRange = 5f..30f,
                    steps = 24,
                    modifier = Modifier.fillMaxWidth()
                )
                Text(
                    text = "От 5 до 30 тараканов одновременно",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.secondary
                )
            }
        }

        Card(
            modifier = Modifier.fillMaxWidth(),
            elevation = CardDefaults.cardElevation(2.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "Интервал бонусов: ${bonusInterval.toInt()} сек",
                    style = MaterialTheme.typography.titleMedium
                )
                Slider(
                    value = bonusInterval,
                    onValueChange = { bonusInterval = it },
                    valueRange = 3f..15f,
                    steps = 11,
                    modifier = Modifier.fillMaxWidth()
                )
                Text(
                    text = "От 3 до 15 секунд между бонусами",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.secondary
                )
            }
        }

        Card(
            modifier = Modifier.fillMaxWidth(),
            elevation = CardDefaults.cardElevation(2.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "Длительность раунда: ${roundDuration.toInt()} сек",
                    style = MaterialTheme.typography.titleMedium
                )
                Slider(
                    value = roundDuration,
                    onValueChange = { roundDuration = it },
                    valueRange = 30f..180f,
                    steps = 14,
                    modifier = Modifier.fillMaxWidth()
                )
                Text(
                    text = "От 30 секунд до 3 минут",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.secondary
                )
            }
        }

        Button(
            onClick = { },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Сохранить настройки")
        }
    }
}