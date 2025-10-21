package com.example.beetles

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.example.beetles.ui.AppNavigation
import com.example.beetles.ui.theme.BeetlesTheme
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            BeetlesTheme(darkTheme = true) {
                AppNavigation()
            }
        }
    }
}