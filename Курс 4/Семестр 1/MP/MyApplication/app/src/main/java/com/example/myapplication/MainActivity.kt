package com.example.myapplication

import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.Spinner
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import com.google.android.material.switchmaterial.SwitchMaterial

class MainActivity : AppCompatActivity() {
    // Variable theme
    private lateinit var switch: SwitchMaterial

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_registration)
//        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
//            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
//            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
//            insets
//        }

        // Spinner
        val spinner: Spinner = findViewById(R.id.spCourse)
        ArrayAdapter.createFromResource(
            this,
            R.array.courses,
            android.R.layout.simple_spinner_item
        ).also { adapter ->
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            spinner.adapter = adapter
        }

        // Theme
        switch = findViewById(R.id.themeSwitch)
        switch.setOnClickListener {
            if (switch.isChecked) {
                AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_YES)
            } else {
                AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_NO)
            }
        }

        // Date Picker
        findViewById<Button>(R.id.btnBirthDate).setOnClickListener {
            val datePickerFragment = DatePickerFragment()
            datePickerFragment.show(supportFragmentManager, "datePicker")
        }

        // Submit Button
        val btnSubmit: Button = findViewById(R.id.btnSubmit)
        btnSubmit.setOnClickListener {
            val birthDateText = findViewById<Button>(R.id.btnBirthDate).text.toString()
            val fullName = findViewById<EditText>(R.id.etFullName).text.toString()
            if (birthDateText != "Выберите дату" && birthDateText.isNotEmpty() && fullName.isNotEmpty()) {
                btnSubmit.isEnabled = true
                val day = birthDateText.substring(0, 2).toIntOrNull() ?: return@setOnClickListener
                val month = birthDateText.substring(3, 5).toIntOrNull() ?: return@setOnClickListener
                val zodiacSign = getZodiacSign(month, day)
                Toast.makeText(this, "Ваш знак зодиака: $zodiacSign", Toast.LENGTH_LONG).show()
            } else {
                when {
                    fullName.isEmpty() && (birthDateText == "Выберите дату" || birthDateText.isEmpty()) ->
                        Toast.makeText(this, "Введите имя и выберите дату рождения", Toast.LENGTH_SHORT).show()

                    fullName.isEmpty() ->
                        Toast.makeText(this, "Введите имя", Toast.LENGTH_SHORT).show()

                    birthDateText == "Выберите дату" || birthDateText.isEmpty() ->
                        Toast.makeText(this, "Выберите дату рождения", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }
    private fun getZodiacSign(month: Int, day: Int): String {
        return when {
            (month == 3 && day >= 21) || (month == 4 && day <= 19) -> "Овен"
            (month == 4 && day >= 20) || (month == 5 && day <= 20) -> "Телец"
            (month == 5 && day >= 21) || (month == 6 && day <= 20) -> "Близнецы"
            (month == 6 && day >= 21) || (month == 7 && day <= 22) -> "Рак"
            (month == 7 && day >= 23) || (month == 8 && day <= 22) -> "Лев"
            (month == 8 && day >= 23) || (month == 9 && day <= 22) -> "Дева"
            (month == 9 && day >= 23) || (month == 10 && day <= 22) -> "Весы"
            (month == 10 && day >= 23) || (month == 11 && day <= 21) -> "Скорпион"
            (month == 11 && day >= 22) || (month == 12 && day <= 21) -> "Стрелец"
            (month == 12 && day >= 22) || (month == 1 && day <= 19) -> "Козерог"
            (month == 1 && day >= 20) || (month == 2 && day <= 18) -> "Водолей"
            (month == 2 && day >= 19) || (month == 3 && day <= 20) -> "Рыбы"
            else -> "Неопределённый знак"
        }
    }
}