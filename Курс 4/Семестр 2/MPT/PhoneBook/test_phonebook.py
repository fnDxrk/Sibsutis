import unittest
import json
import os
import tempfile
from unittest.mock import patch, MagicMock
import tkinter as tk
from phonebook import Abonent, AbonentList, FileManager, PhonebookGUI


class TestAbonent(unittest.TestCase):
    
    def test_abonent_creation(self):
        abonent = Abonent("Иванов Иван", "+7-123-456-78-90")
        self.assertEqual(abonent.full_name, "Иванов Иван")
        self.assertEqual(abonent.phone, "+7-123-456-78-90")
    
    def test_abonent_strip_whitespace(self):
        abonent = Abonent("  Петров Петр  ", "  +7-999-888-77-66  ")
        self.assertEqual(abonent.full_name, "Петров Петр")
        self.assertEqual(abonent.phone, "+7-999-888-77-66")
    
    def test_to_dict(self):
        abonent = Abonent("Сидоров Сидор", "+7-111-222-33-44")
        expected = {"full_name": "Сидоров Сидор", "phone": "+7-111-222-33-44"}
        self.assertEqual(abonent.to_dict(), expected)
    
    def test_from_dict(self):
        data = {"full_name": "Кузнецов Кузьма", "phone": "+7-555-666-77-88"}
        abonent = Abonent.from_dict(data)
        self.assertEqual(abonent.full_name, "Кузнецов Кузьма")
        self.assertEqual(abonent.phone, "+7-555-666-77-88")
    
    def test_from_dict_missing_fields(self):
        data = {}
        abonent = Abonent.from_dict(data)
        self.assertEqual(abonent.full_name, "")
        self.assertEqual(abonent.phone, "")


class TestAbonentList(unittest.TestCase):
    
    def setUp(self):
        self.ab_list = AbonentList()
        self.ab1 = Abonent("Иванов Иван", "111")
        self.ab2 = Abonent("Петров Петр", "222")
        self.ab3 = Abonent("Сидоров Сидор", "333")
    
    def test_add_abonent(self):
        self.ab_list.add(self.ab1)
        self.assertEqual(len(self.ab_list.to_list()), 1)
        self.assertEqual(self.ab_list.to_list()[0].full_name, "Иванов Иван")
    
    def test_add_multiple_abonents(self):
        self.ab_list.add(self.ab1)
        self.ab_list.add(self.ab2)
        self.ab_list.add(self.ab3)
        self.assertEqual(len(self.ab_list.to_list()), 3)
    
    def test_auto_sort_on_add(self):
        self.ab_list.add(self.ab2)
        self.ab_list.add(self.ab1)
        self.ab_list.add(self.ab3)
        
        names = [a.full_name for a in self.ab_list.to_list()]
        expected = ["Иванов Иван", "Петров Петр", "Сидоров Сидор"]
        self.assertEqual(names, expected)
    
    def test_delete_abonent(self):
        self.ab_list.add(self.ab1)
        self.ab_list.add(self.ab2)
        self.ab_list.delete(0)
        self.assertEqual(len(self.ab_list.to_list()), 1)
        self.assertEqual(self.ab_list.to_list()[0].full_name, "Петров Петр")
    
    def test_delete_invalid_index(self):
        self.ab_list.add(self.ab1)
        self.ab_list.delete(10)
        self.assertEqual(len(self.ab_list.to_list()), 1)
        self.ab_list.delete(-1)
        self.assertEqual(len(self.ab_list.to_list()), 1)
    
    def test_update_abonent(self):
        self.ab_list.add(self.ab1)
        new_abonent = Abonent("Новый Имя", "999")
        self.ab_list.update(0, new_abonent)
        self.assertEqual(self.ab_list.to_list()[0].full_name, "Новый Имя")
        self.assertEqual(self.ab_list.to_list()[0].phone, "999")
    
    def test_update_with_sort(self):
        self.ab_list.add(self.ab1)
        self.ab_list.add(self.ab3)
        
        new_abonent = Abonent("Абаков Абакум", "444")
        self.ab_list.update(1, new_abonent)
        
        names = [a.full_name for a in self.ab_list.to_list()]
        expected = ["Абаков Абакум", "Иванов Иван"]
        self.assertEqual(names, expected)
    
    def test_update_invalid_index(self):
        self.ab_list.add(self.ab1)
        new_abonent = Abonent("Новый", "999")
        self.ab_list.update(10, new_abonent)
        self.assertEqual(len(self.ab_list.to_list()), 1)
        self.assertEqual(self.ab_list.to_list()[0].full_name, "Иванов Иван")
    
    def test_clear(self):
        self.ab_list.add(self.ab1)
        self.ab_list.add(self.ab2)
        self.ab_list.clear()
        self.assertEqual(len(self.ab_list.to_list()), 0)
    
    def test_search(self):
        self.ab_list.add(self.ab1)
        self.ab_list.add(self.ab2)
        self.ab_list.add(Abonent("Иванова Мария", "444"))
        
        results = self.ab_list.search("иван")
        self.assertEqual(len(results), 2)
        
        results = self.ab_list.search("петр")
        self.assertEqual(len(results), 1)
        self.assertEqual(self.ab_list.to_list()[results[0]].full_name, "Петров Петр")
    
    def test_search_case_insensitive(self):
        self.ab_list.add(Abonent("ИВАНОВ ИВАН", "111"))
        self.ab_list.add(Abonent("иванов петр", "222"))
        
        results = self.ab_list.search("иванов")
        self.assertEqual(len(results), 2)
    
    def test_search_empty(self):
        self.ab_list.add(self.ab1)
        results = self.ab_list.search("")
        self.assertEqual(len(results), 1)
        results = self.ab_list.search("   ")
        self.assertEqual(len(results), 1)
    
    def test_sort_by_name(self):
        unsorted_list = [self.ab2, self.ab3, self.ab1]
        for a in unsorted_list:
            self.ab_list.add(a)
        
        names = [a.full_name for a in self.ab_list.to_list()]
        expected = ["Иванов Иван", "Петров Петр", "Сидоров Сидор"]
        self.assertEqual(names, expected)
    
    def test_to_json(self):
        self.ab_list.add(self.ab1)
        self.ab_list.add(self.ab2)
        
        json_data = self.ab_list.to_json()
        expected = [
            {"full_name": "Иванов Иван", "phone": "111"},
            {"full_name": "Петров Петр", "phone": "222"}
        ]
        self.assertEqual(json_data, expected)
    
    def test_from_json(self):
        data = [
            {"full_name": "Тестовый Тест", "phone": "123"},
            {"full_name": "Пример Примеров", "phone": "456"}
        ]
        self.ab_list.from_json(data)
        
        self.assertEqual(len(self.ab_list.to_list()), 2)
        self.assertEqual(self.ab_list.to_list()[0].full_name, "Пример Примеров")
        self.assertEqual(self.ab_list.to_list()[1].full_name, "Тестовый Тест")


class TestFileManager(unittest.TestCase):
    
    def setUp(self):
        self.ab_list = AbonentList()
        self.ab_list.add(Abonent("Иванов Иван", "111"))
        self.ab_list.add(Abonent("Петров Петр", "222"))
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.temp_file.close()
        self.filename = self.temp_file.name
    
    def tearDown(self):
        if os.path.exists(self.filename):
            os.unlink(self.filename)
    
    def test_save_phonebook(self):
        success, error = FileManager.save_phonebook(self.ab_list, self.filename)
        self.assertTrue(success)
        self.assertIsNone(error)
        
        with open(self.filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['full_name'], "Иванов Иван")
        self.assertEqual(data[1]['full_name'], "Петров Петр")
    
    def test_save_phonebook_invalid_path(self):
        success, error = FileManager.save_phonebook(self.ab_list, "/invalid/path/file.json")
        self.assertFalse(success)
        self.assertIsNotNone(error)
        self.assertIn("Ошибка записи", error)
    
    def test_load_phonebook(self):
        FileManager.save_phonebook(self.ab_list, self.filename)
        
        new_list = AbonentList()
        success, error = FileManager.load_phonebook(new_list, self.filename)
        
        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertEqual(len(new_list.to_list()), 2)
        self.assertEqual(new_list.to_list()[0].full_name, "Иванов Иван")
        self.assertEqual(new_list.to_list()[1].full_name, "Петров Петр")
    
    def test_load_phonebook_nonexistent_file(self):
        new_list = AbonentList()
        success, error = FileManager.load_phonebook(new_list, "nonexistent.json")
        
        self.assertFalse(success)
        self.assertIsNotNone(error)
        self.assertIn("Ошибка чтения", error)
    
    def test_load_phonebook_invalid_json(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write("{invalid json}")
        
        new_list = AbonentList()
        success, error = FileManager.load_phonebook(new_list, self.filename)
        
        self.assertFalse(success)
        self.assertIsNotNone(error)
        self.assertIn("Ошибка чтения", error)


class TestPhonebookGUI(unittest.TestCase):
    
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = PhonebookGUI(self.root)
    
    def tearDown(self):
        self.root.destroy()
    
    def test_initial_state(self):
        self.assertEqual(len(self.app.abonent_list.to_list()), 0)
        self.assertEqual(self.app.status_var.get(), "Всего записей: 0")
    
    def test_add_abonent(self):
        self.app.entry_name.insert(0, "Тестовый Тест")
        self.app.entry_phone.insert(0, "123456")
        
        self.app.add_abonent()
        
        self.assertEqual(len(self.app.abonent_list.to_list()), 1)
        self.assertEqual(self.app.abonent_list.to_list()[0].full_name, "Тестовый Тест")
        self.assertEqual(self.app.entry_name.get(), "")
        self.assertEqual(self.app.entry_phone.get(), "")
    
    def test_add_abonent_empty_fields(self):
        with patch('tkinter.messagebox.showwarning') as mock_showwarning:
            self.app.add_abonent()
            mock_showwarning.assert_called_once()
        
        self.assertEqual(len(self.app.abonent_list.to_list()), 0)
    
    @patch('tkinter.messagebox.askyesno')
    def test_delete_abonent(self, mock_askyesno):
        mock_askyesno.return_value = True
        
        self.app.abonent_list.add(Abonent("Тестовый", "123"))
        self.app._refresh_tree()
        
        self.app.tree.selection_set("0")
        self.app.delete_abonent()
        
        self.assertEqual(len(self.app.abonent_list.to_list()), 0)
    
    @patch('tkinter.messagebox.askyesno')
    def test_delete_abonent_cancel(self, mock_askyesno):
        mock_askyesno.return_value = False
        
        self.app.abonent_list.add(Abonent("Тестовый", "123"))
        self.app._refresh_tree()
        
        self.app.tree.selection_set("0")
        self.app.delete_abonent()
        
        self.assertEqual(len(self.app.abonent_list.to_list()), 1)
    
    def test_search_functionality(self):
        self.app.abonent_list.add(Abonent("Иванов Иван", "111"))
        self.app.abonent_list.add(Abonent("Петров Петр", "222"))
        self.app.abonent_list.add(Abonent("Иванова Мария", "333"))
        self.app._refresh_tree()
        
        self.app.entry_search.insert(0, "иван")
        self.app.search_abonent()
        
        selected = self.app.tree.selection()
        self.assertEqual(len(selected), 2)
    
    def test_reset_search(self):
        self.app.abonent_list.add(Abonent("Иванов Иван", "111"))
        self.app._refresh_tree()
        
        self.app.tree.selection_set("0")
        self.app.reset_search()
        
        self.assertEqual(len(self.app.tree.selection()), 0)
        self.assertEqual(self.app.entry_search.get(), "")
    
    def test_on_tree_select(self):
        self.app.abonent_list.add(Abonent("Тестовый Тест", "123456"))
        self.app._refresh_tree()
        
        mock_event = MagicMock()
        self.app.tree.selection_set("0")
        self.app.on_tree_select(mock_event)
        
        self.assertEqual(self.app.entry_name.get(), "Тестовый Тест")
        self.assertEqual(self.app.entry_phone.get(), "123456")
    
    def test_edit_abonent(self):
        self.app.abonent_list.add(Abonent("Старое Имя", "111"))
        self.app._refresh_tree()
        
        self.app.tree.selection_set("0")
        self.app.entry_name.delete(0, tk.END)
        self.app.entry_name.insert(0, "Новое Имя")
        self.app.entry_phone.delete(0, tk.END)
        self.app.entry_phone.insert(0, "222")
        
        self.app.edit_abonent()
        
        self.assertEqual(self.app.abonent_list.to_list()[0].full_name, "Новое Имя")
        self.assertEqual(self.app.abonent_list.to_list()[0].phone, "222")
    
    @patch('tkinter.messagebox.askyesno')
    def test_clear_book(self, mock_askyesno):
        mock_askyesno.return_value = True
        
        self.app.abonent_list.add(Abonent("Тестовый", "123"))
        self.app.abonent_list.add(Abonent("Тестовый2", "456"))
        
        self.app.clear_book()
        
        self.assertEqual(len(self.app.abonent_list.to_list()), 0)
    
    @patch('tkinter.filedialog.asksaveasfilename')
    @patch('phonebook.FileManager.save_phonebook')
    def test_save_book(self, mock_save, mock_filename):
        mock_filename.return_value = "test.json"
        mock_save.return_value = (True, None)
        
        self.app.save_book()
        
        mock_save.assert_called_once()
    
    @patch('tkinter.filedialog.askopenfilename')
    @patch('phonebook.FileManager.load_phonebook')
    def test_load_book(self, mock_load, mock_filename):
        mock_filename.return_value = "test.json"
        mock_load.return_value = (True, None)
        
        self.app.load_book()
        
        mock_load.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)
