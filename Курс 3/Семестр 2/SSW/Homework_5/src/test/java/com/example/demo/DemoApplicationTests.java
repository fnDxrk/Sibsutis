package com.example.demo;

import com.example.demo.User;
import com.example.demo.UserRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest
class DemoApplicationTests {

	@Autowired
	private UserRepository userRepository;

	@Test
	void testCreateAndFindUser() {
		User user = new User(null, "testuser", "test@example.com");
		User savedUser = userRepository.save(user);

		assertThat(savedUser.getId()).isNotNull();
		assertThat(savedUser.getUsername()).isEqualTo("testuser");
	}
}
