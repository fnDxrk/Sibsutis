package com.example.demo;

import jakarta.persistence.*;

@Entity
@Table(name = "car")
public class Car {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String manufacturer;

    private Float velocity;

    private String kind;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "person_id", nullable = false)
    private Person person;

    public Car() {}

    public Car(String manufacturer, Float velocity, String kind, Person person) {
        this.manufacturer = manufacturer;
        this.velocity = velocity;
        this.kind = kind;
        this.person = person;
    }

    public Long getId() { return id; }

    public String getManufacturer() { return manufacturer; }
    public void setManufacturer(String manufacturer) { this.manufacturer = manufacturer; }

    public Float getVelocity() { return velocity; }
    public void setVelocity(Float velocity) { this.velocity = velocity; }

    public String getKind() { return kind; }
    public void setKind(String kind) { this.kind = kind; }

    public Person getPerson() { return person; }
    public void setPerson(Person person) { this.person = person; }
}
