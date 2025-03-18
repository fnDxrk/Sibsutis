package com.example.demo;

import jakarta.persistence.*;  //привязываем класс к бд

@Entity
@Table(name = "users") //имя таблицы
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)  //генерируем автоматически id
    private Long id;

    private String username;
    private String email;

    public User() {
    }

    public User(Long id, String username, String email) {
        this.id = id;
        this.username = username;
        this.email = email;
    }

    public Long getId() {
        return id;
    }

    public String getUsername() {
        return username;
    }

    public String getEmail() {
        return email;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public void setEmail(String email) {
        this.email = email;
    }
}
