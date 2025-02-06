#include <elf.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Функция для чтения заголовка ELF
void read_elf_header(const char* file, Elf64_Ehdr* ehdr)
{
    int fd = open(file, O_RDONLY);
    if (fd < 0) {
        perror("open");
        exit(EXIT_FAILURE);
    }
    if (read(fd, ehdr, sizeof(Elf64_Ehdr)) != sizeof(Elf64_Ehdr)) {
        perror("read");
        close(fd);
        exit(EXIT_FAILURE);
    }
    close(fd);
}

// Функция для чтения таблицы секций
void read_section_headers(const char* file, Elf64_Ehdr* ehdr,
    Elf64_Shdr** shdrs, char** shstrtab)
{
    int fd = open(file, O_RDONLY);
    if (fd < 0) {
        perror("open");
        exit(EXIT_FAILURE);
    }

    *shdrs = malloc(ehdr->e_shnum * sizeof(Elf64_Shdr));
    if (*shdrs == NULL) {
        perror("malloc");
        close(fd);
        exit(EXIT_FAILURE);
    }

    lseek(fd, ehdr->e_shoff, SEEK_SET);
    read(fd, *shdrs, ehdr->e_shnum * sizeof(Elf64_Shdr));

    // Чтение строки секции
    Elf64_Shdr* shstrtab_hdr = &(*shdrs)[ehdr->e_shstrndx];
    *shstrtab = malloc(shstrtab_hdr->sh_size);
    if (*shstrtab == NULL) {
        perror("malloc");
        free(*shdrs);
        close(fd);
        exit(EXIT_FAILURE);
    }

    lseek(fd, shstrtab_hdr->sh_offset, SEEK_SET);
    read(fd, *shstrtab, shstrtab_hdr->sh_size);

    close(fd);
}

// Функция для получения и отображения имени секции по индексу
void print_section_names(Elf64_Shdr* shdrs, char* shstrtab, Elf64_Ehdr* ehdr)
{
    printf("Section header string table index: %d\n", ehdr->e_shstrndx);
    printf("Section names:\n");

    for (int i = 0; i < ehdr->e_shnum; i++) {
        printf("Section %d: %s\n", i, &shstrtab[shdrs[i].sh_name]);
    }
}

// Функция для извлечения и отображения имен экспортируемых функций
void print_exported_functions(const char* file, Elf64_Ehdr* ehdr,
    Elf64_Shdr* shdrs, char* shstrtab)
{
    Elf64_Shdr* dynsym_hdr = NULL;
    Elf64_Shdr* dynstr_hdr = NULL;

    // Найти секции динамических символов и строк
    for (int i = 0; i < ehdr->e_shnum; i++) {
        if (shdrs[i].sh_type == SHT_DYNSYM) {
            dynsym_hdr = &shdrs[i];
        } else if (shdrs[i].sh_type == SHT_STRTAB && strcmp(&shstrtab[shdrs[i].sh_name], ".dynstr") == 0) {
            dynstr_hdr = &shdrs[i];
        }
    }

    if (dynsym_hdr == NULL || dynstr_hdr == NULL) {
        fprintf(stderr, "Required sections not found\n");
        return;
    }

    int num_symbols = dynsym_hdr->sh_size / dynsym_hdr->sh_entsize;
    Elf64_Sym* symbols = malloc(dynsym_hdr->sh_size);
    if (symbols == NULL) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    int fd = open(file, O_RDONLY);
    lseek(fd, dynsym_hdr->sh_offset, SEEK_SET);
    read(fd, symbols, dynsym_hdr->sh_size);

    char* dynstr_tab = malloc(dynstr_hdr->sh_size);
    if (dynstr_tab == NULL) {
        perror("malloc");
        free(symbols);
        close(fd);
        exit(EXIT_FAILURE);
    }

    lseek(fd, dynstr_hdr->sh_offset, SEEK_SET);
    read(fd, dynstr_tab, dynstr_hdr->sh_size);
    close(fd);

    printf("Exported functions:\n");
    for (int i = 0; i < num_symbols; i++) {
        if (ELF64_ST_TYPE(symbols[i].st_info) == STT_FUNC && symbols[i].st_value != 0) {
            printf("  %s\n", &dynstr_tab[symbols[i].st_name]);
        }
    }

    free(symbols);
    free(dynstr_tab);
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <elf_file>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const char* file = argv[1];
    Elf64_Ehdr ehdr;
    Elf64_Shdr* shdrs = NULL;
    char* shstrtab = NULL;

    read_elf_header(file, &ehdr);
    read_section_headers(file, &ehdr, &shdrs, &shstrtab);
    print_exported_functions(file, &ehdr, shdrs, shstrtab);

    free(shdrs);
    free(shstrtab);

    return 0;
}
