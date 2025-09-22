# Команды dotnet

## Создание решения
```bash
dotnet new sln -n Labs
```
Создаёт новый файл решения с именем `Labs.sln`.

## Создание консольного проекта
```bash
dotnet new console -n Lab1 -o "Lab 1"
```
Создаёт новый консольный проект с именем `Lab1` в папке `Lab 1`.

## Добавление проекта в решение
```bash
dotnet sln Labs.sln add Lab\ 1/Lab1.csproj
```
Добавляет проект `Lab1` в решение `Labs.sln`.

## Создание тестового проекта
```bash
dotnet new mstest -n "Tests" -o "Lab 1 Tests"
```
Создаёт новый проект MSTest с именем `Tests` в папке `Lab 1 Tests`.

## Добавление проекта в решение
```bash
dotnet sln Labs.sln add Lab\ 1\ Tests/Tests.csproj
```
Добавляет проект `Tests` в решение `Labs.sln`.

## Добавление ссылки на проект
```bash
dotnet add reference "../Lab 1/Lab1.csproj"
```
Добавляет ссылку на проект `Lab1` в текущий проект (выполняется в папке проекта `Tests`).

## Сборка решения
```bash
dotnet build Labs.sln
```
Выполняет сборку всех проектов в решении `Labs.sln`.

## Запуск проекта
```bash
dotnet run --project 'Lab 1'
```
Запускает консольное приложение Lab1 из папки Lab 1.

## Запуск тестов
```bash
dotnet test Labs.sln
```
Запускает тесты для всех тестовых проектов в решении `Labs.sln`.