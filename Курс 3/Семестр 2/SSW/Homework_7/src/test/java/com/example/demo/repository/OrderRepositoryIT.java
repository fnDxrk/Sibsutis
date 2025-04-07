package com.example.demo.repository;

import com.example.demo.model.*;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;

import static org.assertj.core.api.Assertions.assertThat;

@DataJpaTest
@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
@Transactional
class OrderRepositoryIT {

    @Autowired
    private OrderRepository orderRepository;

    @Autowired
    private CustomerRepository customerRepository;  // Добавь зависимость

    @Autowired
    private PaymentRepository paymentRepository;    // Добавь зависимость

    @Test
    void shouldSaveAndRetrieveOrder() {
        // Сначала сохраняем Customer
        Customer customer = new Customer("Test Customer", new Address("Moscow", "Lenina", "123456"));
        customer = customerRepository.save(customer);  // 💾 Сохраняем в БД

        // Затем сохраняем Payment (Cash)
        Payment payment = new Cash(100.0f, "PAID", 100.0f);
        payment = paymentRepository.save(payment);  // 💾 Сохраняем в БД

        // Теперь создаём Order с уже сохранёнными сущностями
        Order order = new Order();
        order.setDate(LocalDateTime.now());
        order.setStatus("CREATED");
        order.setCustomer(customer);
        order.setPayment(payment);

        // Сохраняем Order
        Order saved = orderRepository.save(order);

        // Проверяем результат
        assertThat(saved.getId()).isNotNull();
        assertThat(orderRepository.findById(saved.getId()))
                .isPresent()
                .hasValueSatisfying(o -> assertThat(o.getStatus()).isEqualTo("CREATED"));
    }
}
