package com.example.demo.repository;

import com.example.demo.model.Order;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface OrderRepository extends JpaRepository<Order, Long>, JpaSpecificationExecutor<Order> {

    // Дополнительные методы запросов (если нужны)
    List<Order> findByCustomerNameContainingIgnoreCase(String customerName);
    List<Order> findByStatus(String status);
    List<Order> findByCustomerAddressCity(String city);
}