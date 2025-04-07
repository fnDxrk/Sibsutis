package com.example.demo.service;

import com.example.demo.AbstractIntegrationTest;
import com.example.demo.model.*;
import com.example.demo.repository.CustomerRepository;
import com.example.demo.repository.PaymentRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest
@Transactional
class OrderServiceIT extends AbstractIntegrationTest {

    @Autowired
    private OrderService orderService;

    @Autowired
    private CustomerRepository customerRepository; // Добавляем CustomerRepository

    @Autowired
    private PaymentRepository paymentRepository; // Добавляем PaymentRepository

    @Test
    void shouldCreateOrder() {
        // Сначала сохраняем зависимые сущности
        Customer customer = new Customer("Test Customer", new Address("Moscow", "Lenina", "123456"));
        customer = customerRepository.save(customer);  // Сохраняем клиента

        Payment payment = new Cash(100.0f, "PAID", 100.0f);
        payment = paymentRepository.save(payment);    // Сохраняем платеж

        // Теперь создаем заказ с сохраненными сущностями
        Order order = new Order();
        order.setDate(LocalDateTime.now());
        order.setStatus("CREATED");
        order.setCustomer(customer);
        order.setPayment(payment);

        // Сохраняем заказ
        Order created = orderService.createOrder(order);

        // Проверяем результат
        assertThat(created.getId()).isNotNull();
    }


    @Test
    void shouldFindOrder() {
        Customer customer = new Customer("Test Customer", new Address("Moscow", "Lenina", "123456"));
        customer = customerRepository.save(customer);  // Сохраняем клиента

        Payment payment = new Cash(100.0f, "PAID", 100.0f);
        payment = paymentRepository.save(payment);  // Сохраняем платеж

        Order order = new Order();
        order.setDate(LocalDateTime.now());
        order.setStatus("CREATED");
        order.setCustomer(customer);
        order.setPayment(payment);

        orderService.createOrder(order);

        List<Order> found = orderService.searchOrders("Test", "Moscow", "CREATED");

        assertThat(found)
                .hasSize(1)
                .first()
                .satisfies(o -> {
                    assertThat(o.getStatus()).isEqualTo("CREATED");
                    assertThat(o.getCustomer().getName()).contains("Test");
                });
    }
}
