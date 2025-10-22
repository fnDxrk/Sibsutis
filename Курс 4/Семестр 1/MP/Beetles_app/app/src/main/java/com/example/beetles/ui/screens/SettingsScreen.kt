package com.example.beetles.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.beetles.viewmodel.SettingsViewModel

@Composable
fun SettingsScreen() {
    val context = LocalContext.current
    val viewModel: SettingsViewModel = viewModel(
        factory = androidx.lifecycle.ViewModelProvider.AndroidViewModelFactory.getInstance(
            context.applicationContext as android.app.Application
        )
    )

    val settings by viewModel.settings.collectAsState()

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
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceContainer
            ),
            elevation = CardDefaults.cardElevation(2.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "Скорость игры: ${settings.gameSpeed.toInt()}",
                    style = MaterialTheme.typography.titleMedium
                )
                Slider(
                    value = settings.gameSpeed,
                    onValueChange = { viewModel.updateGameSpeed(it) },
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
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceContainer
            ),
            elevation = CardDefaults.cardElevation(2.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "Макс. количество тараканов: ${settings.maxBeetles}",
                    style = MaterialTheme.typography.titleMedium
                )
                Slider(
                    value = settings.maxBeetles.toFloat(),
                    onValueChange = { viewModel.updateMaxBeetles(it.toInt()) },
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
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceContainer
            ),
            elevation = CardDefaults.cardElevation(2.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "Интервал бонусов: ${settings.bonusInterval} сек",
                    style = MaterialTheme.typography.titleMedium
                )
                Slider(
                    value = settings.bonusInterval.toFloat(),
                    onValueChange = { viewModel.updateBonusInterval(it.toInt()) },
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
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceContainer
            ),
            elevation = CardDefaults.cardElevation(2.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "Длительность раунда: ${settings.roundDuration} сек",
                    style = MaterialTheme.typography.titleMedium
                )
                Slider(
                    value = settings.roundDuration.toFloat(),
                    onValueChange = { viewModel.updateRoundDuration(it.toInt()) },
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
    }
}
