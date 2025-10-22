package com.example.beetles.ui

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AccountBox
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.compose.currentBackStackEntryAsState
import com.example.beetles.ui.screens.AuthorsScreen
import com.example.beetles.ui.screens.RegistrationScreen
import com.example.beetles.ui.screens.RulesScreen
import com.example.beetles.ui.screens.SettingsScreen
import com.example.beetles.ui.screens.GameScreen
import com.example.beetles.viewmodel.GameViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.compose.ui.platform.LocalContext

@Composable
fun AppNavigation() {
    val navController = rememberNavController()
    val context = LocalContext.current
    var selectedItem by remember { mutableIntStateOf(0) }
    val currentRoute = navController.currentBackStackEntryAsState().value?.destination?.route
    val showBottomBar = currentRoute != "game"

    val gameViewModel: GameViewModel = viewModel(
        factory = ViewModelProvider.AndroidViewModelFactory.getInstance(
            context.applicationContext as android.app.Application
        )
    )

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        bottomBar = {
            if (showBottomBar) {
                NavigationBar {
                    NavigationBarItem(
                        icon = { Icon(Icons.Default.Person, contentDescription = "Регистрация") },
                        label = { Text("Регистрация") },
                        selected = selectedItem == 0,
                        onClick = {
                            selectedItem = 0
                            navController.navigate("registration") {
                                popUpTo("registration") { inclusive = true }
                                launchSingleTop = true
                            }
                        }
                    )

                    NavigationBarItem(
                        icon = { Icon(Icons.Default.Info, contentDescription = "Правила") },
                        label = { Text("Правила") },
                        selected = selectedItem == 1,
                        onClick = {
                            selectedItem = 1
                            navController.navigate("rules") {
                                launchSingleTop = true
                            }
                        }
                    )

                    NavigationBarItem(
                        icon = { Icon(Icons.Default.AccountBox, contentDescription = "Автор") },
                        label = { Text("Автор") },
                        selected = selectedItem == 2,
                        onClick = {
                            selectedItem = 2
                            navController.navigate("authors") {
                                launchSingleTop = true
                            }
                        }
                    )

                    NavigationBarItem(
                        icon = { Icon(Icons.Default.Settings, contentDescription = "Настройки") },
                        label = { Text("Настройки") },
                        selected = selectedItem == 3,
                        onClick = {
                            selectedItem = 3
                            navController.navigate("settings") {
                                launchSingleTop = true
                            }
                        }
                    )
                }
            }
        }
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = "registration",
            modifier = Modifier.padding(innerPadding)
        ) {
            composable("registration") {
                RegistrationScreen(navController = navController)
            }

            composable("rules") {
                RulesScreen()
            }

            composable("authors") {
                AuthorsScreen()
            }

            composable("settings") {
                SettingsScreen()
            }

            composable("game") {
                GameScreen(
                    viewModel = gameViewModel,
                    onBackClick = {
                        gameViewModel.resetGame()
                        navController.popBackStack()
                    }
                )
            }
        }
    }
}
