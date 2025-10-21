package com.example.beetles.ui.screens

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import java.text.SimpleDateFormat
import java.util.*

import com.example.beetles.data.PlayerData
import com.example.beetles.utils.calculateZodiacSign
import com.example.beetles.utils.getZodiacImageResource

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RegistrationScreen(modifier: Modifier = Modifier) {
    var fullName by remember { mutableStateOf("") }
    var selectedGender by remember { mutableStateOf("") }
    var selectedCourse by remember { mutableStateOf("") }
    var difficulty by remember { mutableFloatStateOf(1f) }
    var selectedDate by remember { mutableLongStateOf(System.currentTimeMillis()) }
    var showDatePicker by remember { mutableStateOf(false) }
    var playerData by remember { mutableStateOf<PlayerData?>(null) }
    var showZodiacDialog by remember { mutableStateOf(false) }
    var dateSelected by remember { mutableStateOf(false) }

    val dateFormatter = SimpleDateFormat("dd.MM.yyyy", Locale.getDefault())
    val calendar = Calendar.getInstance().apply { timeInMillis = selectedDate }

    val isFormValid = fullName.isNotBlank() &&
            selectedGender.isNotBlank() &&
            selectedCourse.isNotBlank() &&
            dateSelected

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text(
            text = "Регистрация игрока",
            modifier = Modifier.fillMaxWidth(),
            textAlign = TextAlign.Center,
            style = MaterialTheme.typography.headlineMedium
        )

        OutlinedTextField(
            value = fullName,
            onValueChange = { fullName = it },
            label = { Text("ФИО") },
            placeholder = { Text("Введите ваше ФИО") },
            modifier = Modifier.fillMaxWidth()
        )

        Text(
            text = "Пол:",
            style = MaterialTheme.typography.titleMedium
        )

        val genderOptions = listOf("Мужской", "Женский")

        Column {
            genderOptions.forEach { gender ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .selectable(
                            selected = (gender == selectedGender),
                            onClick = { selectedGender = gender }
                        )
                        .padding(vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    RadioButton(
                        selected = (gender == selectedGender),
                        onClick = { selectedGender = gender }
                    )
                    Text(
                        text = gender,
                        style = MaterialTheme.typography.bodyLarge,
                        modifier = Modifier.padding(start = 8.dp)
                    )
                }
            }
        }

        Text(
            text = "Курс:",
            style = MaterialTheme.typography.titleMedium
        )

        var expandedCourse by remember { mutableStateOf(false) }
        val courses = listOf("1 курс", "2 курс", "3 курс", "4 курс")

        ExposedDropdownMenuBox(
            expanded = expandedCourse,
            onExpandedChange = { expandedCourse = !expandedCourse }
        ) {
            OutlinedTextField(
                value = selectedCourse.ifEmpty { "Не выбран" },
                onValueChange = {},
                readOnly = true,
                label = { Text("Выберите курс") },
                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expandedCourse) },
                modifier = Modifier
                    .menuAnchor()
                    .fillMaxWidth()
            )

            ExposedDropdownMenu(
                expanded = expandedCourse,
                onDismissRequest = { expandedCourse = false }
            ) {
                courses.forEach { course ->
                    DropdownMenuItem(
                        text = { Text(course) },
                        onClick = {
                            selectedCourse = course
                            expandedCourse = false
                        }
                    )
                }
            }
        }

        Text(
            text = "Уровень сложности: ${difficulty.toInt()}",
            style = MaterialTheme.typography.titleMedium
        )
        Slider(
            value = difficulty,
            onValueChange = { difficulty = it },
            valueRange = 1f..3f,
            steps = 1,
            modifier = Modifier.fillMaxWidth()
        )

        Text(
            text = "Дата рождения:",
            style = MaterialTheme.typography.titleMedium
        )

        Button(
            onClick = { showDatePicker = true },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(
                if (dateSelected) {
                    "Выбрать дату: ${dateFormatter.format(Date(selectedDate))}"
                } else {
                    "Выбрать дату рождения"
                }
            )
        }

        if (showDatePicker) {
            val datePickerState = rememberDatePickerState(
                initialSelectedDateMillis = selectedDate
            )

            DatePickerDialog(
                onDismissRequest = { showDatePicker = false },
                confirmButton = {
                    TextButton(onClick = {
                        datePickerState.selectedDateMillis?.let {
                            selectedDate = it
                            dateSelected = true
                        }
                        showDatePicker = false
                    }) {
                        Text("OK")
                    }
                },
                dismissButton = {
                    TextButton(onClick = { showDatePicker = false }) {
                        Text("Отмена")
                    }
                }
            ) {
                DatePicker(state = datePickerState)
            }
        }

        Button(
            onClick = {
                val day = calendar.get(Calendar.DAY_OF_MONTH)
                val month = calendar.get(Calendar.MONTH) + 1
                val zodiac = calculateZodiacSign(day, month)

                playerData = PlayerData(
                    fullName = fullName,
                    gender = selectedGender,
                    course = selectedCourse,
                    difficulty = difficulty.toInt(),
                    birthDate = dateFormatter.format(Date(selectedDate)),
                    zodiacSign = zodiac
                )

                showZodiacDialog = true
            },
            enabled = isFormValid,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Зарегистрировать")
        }
    }

    if (showZodiacDialog && playerData != null) {
        AlertDialog(
            onDismissRequest = { showZodiacDialog = false },
            title = {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "Ваш знак зодиака:",
                        style = MaterialTheme.typography.titleMedium,
                        textAlign = TextAlign.Center
                    )
                    Text(
                        text = playerData!!.zodiacSign,
                        style = MaterialTheme.typography.headlineLarge,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(top = 8.dp)
                    )
                }
            },
            text = {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Image(
                        painter = painterResource(id = getZodiacImageResource(playerData!!.zodiacSign)),
                        contentDescription = "Знак зодиака ${playerData!!.zodiacSign}",
                        modifier = Modifier.size(128.dp)
                    )
                }
            },
            confirmButton = {
                TextButton(onClick = { showZodiacDialog = false }) {
                    Text("Закрыть")
                }
            }
        )
    }
}