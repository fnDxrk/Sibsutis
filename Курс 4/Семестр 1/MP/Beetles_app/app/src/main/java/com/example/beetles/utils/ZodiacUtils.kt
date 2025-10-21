package com.example.beetles.utils

import com.example.beetles.R

fun calculateZodiacSign(day: Int, month: Int): String {
    return when (month) {
        1 -> if (day <= 19) "Козерог" else "Водолей"
        2 -> if (day <= 18) "Водолей" else "Рыбы"
        3 -> if (day <= 20) "Рыбы" else "Овен"
        4 -> if (day <= 19) "Овен" else "Телец"
        5 -> if (day <= 20) "Телец" else "Близнецы"
        6 -> if (day <= 20) "Близнецы" else "Рак"
        7 -> if (day <= 22) "Рак" else "Лев"
        8 -> if (day <= 22) "Лев" else "Дева"
        9 -> if (day <= 22) "Дева" else "Весы"
        10 -> if (day <= 22) "Весы" else "Скорпион"
        11 -> if (day <= 21) "Скорпион" else "Стрелец"
        12 -> if (day <= 21) "Стрелец" else "Козерог"
        else -> "Неизвестно"
    }
}

fun getZodiacImageResource(zodiacSign: String): Int {
    return when {
        zodiacSign.contains("Овен") -> R.drawable.aries
        zodiacSign.contains("Телец") -> R.drawable.taurus
        zodiacSign.contains("Близнецы") -> R.drawable.gemini
        zodiacSign.contains("Рак") -> R.drawable.cancer
        zodiacSign.contains("Лев") -> R.drawable.leo
        zodiacSign.contains("Дева") -> R.drawable.virgo
        zodiacSign.contains("Весы") -> R.drawable.libra
        zodiacSign.contains("Скорпион") -> R.drawable.scorpio
        zodiacSign.contains("Стрелец") -> R.drawable.sagittarius
        zodiacSign.contains("Козерог") -> R.drawable.capricorn
        zodiacSign.contains("Водолей") -> R.drawable.aquarius
        zodiacSign.contains("Рыбы") -> R.drawable.pisces
        else -> R.drawable.aries
    }
}