package com.example.demo.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;
import java.util.List;
import java.util.ArrayList;

@Entity
@Table(name = "orders")
public class Order {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private LocalDateTime date;
    private String status;

    // Внешний ключ на Customer с каскадированием
    @ManyToOne
    @JoinColumn(name = "customer_id", nullable = false)  // Убедись, что это не null
    private Customer customer;

    // Внешний ключ на Payment с каскадированием
    @OneToOne(cascade = CascadeType.ALL)
    @JoinColumn(name = "payment_id", nullable = false)
    private Payment payment;

    // Связь с деталями заказа, с каскадированием и удалением сирот
    @OneToMany(mappedBy = "order", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<OrderDetail> orderDetails = new ArrayList<>();

    // Конструкторы
    public Order() {}

    public Order(LocalDateTime date, String status, Customer customer, Payment payment) {
        this.date = date;
        this.status = status;
        this.customer = customer;
        this.payment = payment;
    }

    // Геттеры и сеттеры
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public LocalDateTime getDate() {
        return date;
    }

    public void setDate(LocalDateTime date) {
        this.date = date;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public Customer getCustomer() {
        return customer;
    }

    public void setCustomer(Customer customer) {
        this.customer = customer;
    }

    public Payment getPayment() {
        return payment;
    }

    public void setPayment(Payment payment) {
        this.payment = payment;
    }

    public List<OrderDetail> getOrderDetails() {
        return orderDetails;
    }

    public void setOrderDetails(List<OrderDetail> orderDetails) {
        this.orderDetails = orderDetails;
    }

    public void addOrderDetail(OrderDetail orderDetail) {
        orderDetails.add(orderDetail);
        orderDetail.setOrder(this);
    }

    public void removeOrderDetail(OrderDetail orderDetail) {
        orderDetails.remove(orderDetail);
        orderDetail.setOrder(null);
    }
}
