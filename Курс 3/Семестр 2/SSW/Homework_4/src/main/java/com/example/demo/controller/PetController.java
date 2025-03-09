package com.example.demo.controller;

import com.example.demo.model.Pet;
import com.example.demo.service.PetService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v3")
public class PetController {

    private final PetService petService;

    public PetController(PetService petService) {
        this.petService = petService;
    }

    @PostMapping("/pet")
    public ResponseEntity<Pet> addPet(@RequestBody Pet pet) {
        Pet createdPet = petService.addPet(pet);
        return new ResponseEntity<>(createdPet, HttpStatus.OK);
    }

    @PutMapping("/pet")
    public ResponseEntity<?> updatePet(@RequestBody Pet pet) {
        try {
            Pet updatedPet = petService.updatePet(pet);
            return new ResponseEntity<>(updatedPet, HttpStatus.OK);
        } catch (RuntimeException ex) {
            // Можно более детально обрабатывать ошибки (например, возвращать 400 или 404)
            return new ResponseEntity<>(ex.getMessage(), HttpStatus.NOT_FOUND);
        }
    }

    @GetMapping("/pet/{petId}")
    public ResponseEntity<?> getPetById(@PathVariable Long petId) {
        try {
            Pet pet = petService.getPetById(petId);
            return new ResponseEntity<>(pet, HttpStatus.OK);
        } catch (RuntimeException ex) {
            return new ResponseEntity<>(ex.getMessage(), HttpStatus.NOT_FOUND);
        }
    }

    @DeleteMapping("/pet/{petId}")
    public ResponseEntity<?> deletePet(@PathVariable Long petId) {
        try {
            petService.deletePet(petId);
            return new ResponseEntity<>(HttpStatus.OK);
        } catch (RuntimeException ex) {
            return new ResponseEntity<>(ex.getMessage(), HttpStatus.BAD_REQUEST);
        }
    }
}
