import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

DATA_FILE = "phonebook.json"


class Abonent:
    def __init__(self, full_name: str, phone: str):
        self.full_name = full_name.strip()
        self.phone = phone.strip()

    def to_dict(self):
        return {"full_name": self.full_name, "phone": self.phone}

    @staticmethod
    def from_dict(d: dict):
        return Abonent(d.get("full_name", ""), d.get("phone", ""))


class AbonentList:
    def __init__(self):
        self._abonents: list[Abonent] = []

    def add(self, abonent: Abonent):
        self._abonents.append(abonent)
        self.sort_by_name()

    def delete(self, index: int):
        if 0 <= index < len(self._abonents):
            del self._abonents[index]

    def update(self, index: int, abonent: Abonent):
        if 0 <= index < len(self._abonents):
            self._abonents[index] = abonent
            self.sort_by_name()

    def clear(self):
        self._abonents.clear()

    def search(self, name_part: str):
        name_part = name_part.strip().lower()
        return [i for i, a in enumerate(self._abonents)
                if name_part in a.full_name.lower()]

    def sort_by_name(self):
        self._abonents.sort(key=lambda a: a.full_name.lower())

    def to_list(self):
        return self._abonents

    def to_json(self):
        return [a.to_dict() for a in self._abonents]

    def from_json(self, data):
        self._abonents = [Abonent.from_dict(d) for d in data]
        self.sort_by_name()


class FileManager:
    @staticmethod
    def load_phonebook(ab_list: AbonentList, filename: str):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            ab_list.from_json(data)
            return True, None
        except Exception as e:
            return False, f"Ошибка чтения файла: {e}"

    @staticmethod
    def save_phonebook(ab_list: AbonentList, filename: str):
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(ab_list.to_json(), f, ensure_ascii=False, indent=2)
            return True, None
        except Exception as e:
            return False, f"Ошибка записи файла: {e}"


class PhonebookGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Телефонная книга")
        self.root.geometry("800x550")
        
        self.font_family = "monospace"
        self.font_size = 12
        self.font = (self.font_family, self.font_size)
        self.font_bold = (self.font_family, self.font_size, "bold")
        self.font_small = (self.font_family, 9)
        
        self.colors = {
            'bg_dark': '#1e1e1e',
            'bg_medium': '#2d2d2d',
            'bg_light': '#3c3c3c',
            'fg_light': '#ffffff',
            'fg_dim': '#a0a0a0',
            'accent': '#007acc',
            'accent_hover': '#1c97ea',
            'danger': '#c42b1c',
            'success': '#2e7d32',
            'border': '#555555'
        }
        
        self.root.configure(bg=self.colors['bg_dark'])
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.style.configure('Treeview', 
                            background=self.colors['bg_medium'],
                            foreground=self.colors['fg_light'],
                            fieldbackground=self.colors['bg_medium'],
                            borderwidth=0,
                            font=self.font)
        self.style.map('Treeview', 
                      background=[('selected', self.colors['accent'])])
        
        self.style.configure('Treeview.Heading', 
                            background=self.colors['bg_light'],
                            foreground=self.colors['fg_light'],
                            font=self.font_bold)
        self.style.map('Treeview.Heading',
                      background=[('active', self.colors['accent'])])
        
        self.style.configure('Heading.TLabel', 
                            background=self.colors['bg_dark'],
                            foreground=self.colors['fg_light'],
                            font=self.font_bold)
        
        self.abonent_list = AbonentList()

        self._build_widgets()
        self._refresh_tree()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def show_help(self):
        help_text = """ТЕЛЕФОННАЯ КНИГА v1.0
========================

Как пользоваться программой:

1. Добавление записи:
   - Заполните поля "ФИО" и "Телефон"
   - Нажмите кнопку "Добавить"

2. Изменение записи:
   - Выберите запись в таблице
   - Измените данные в полях ввода
   - Нажмите кнопку "Изменить"

3. Удаление записи:
   - Выберите запись в таблице
   - Нажмите кнопку "Удалить"

4. Поиск:
   - Введите часть имени в поле поиска
   - Нажмите "Найти" для поиска
   - "Сброс" - показать все записи

5. Сохранение и загрузка:
   - "Сохранить" - сохранить книгу в файл
   - "Загрузить" - загрузить книгу из файла

6. Очистка:
   - "Очистить книгу" - удалить все записи

Данные автоматически сортируются по ФИО."""
        
        messagebox.showinfo("Справка", help_text)

    def create_button(self, parent, text, command, color_type='accent', width=None):
        colors = {
            'accent': {'bg': self.colors['accent'], 'hover': self.colors['accent_hover']},
            'danger': {'bg': self.colors['danger'], 'hover': '#a02020'},
            'success': {'bg': self.colors['success'], 'hover': '#1e5f20'}
        }
        color = colors.get(color_type, colors['accent'])
        
        btn = tk.Button(parent, text=text, command=command,
                       bg=color['bg'], fg=self.colors['fg_light'],
                       activebackground=color['hover'],
                       activeforeground=self.colors['fg_light'],
                       relief=tk.FLAT, bd=0, padx=15, pady=5,
                       font=self.font_bold, cursor='hand2')
        
        if width:
            btn.config(width=width)
        
        def on_enter(e):
            btn['background'] = color['hover']
        
        def on_leave(e):
            btn['background'] = color['bg']
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        
        return btn

    def _build_widgets(self):
        menu_bar = tk.Menu(self.root, bg=self.colors['bg_medium'], 
                          fg=self.colors['fg_light'],
                          activebackground=self.colors['accent'],
                          activeforeground=self.colors['fg_light'],
                          font=self.font)
        self.root.config(menu=menu_bar)
        
        help_menu = tk.Menu(menu_bar, tearoff=0, bg=self.colors['bg_medium'],
                           fg=self.colors['fg_light'],
                           activebackground=self.colors['accent'],
                           activeforeground=self.colors['fg_light'],
                           font=self.font)
        menu_bar.add_cascade(label="Справка", menu=help_menu)
        help_menu.add_command(label="О программе", command=self.show_help)
        
        main_container = tk.Frame(self.root, bg=self.colors['bg_dark'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        input_frame = tk.LabelFrame(main_container, text="Данные абонента",
                                   bg=self.colors['bg_dark'],
                                   fg=self.colors['fg_light'],
                                   font=self.font_bold,
                                   padx=10, pady=10)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        input_grid = tk.Frame(input_frame, bg=self.colors['bg_dark'])
        input_grid.pack(fill=tk.X)
        
        tk.Label(input_grid, text="ФИО:", bg=self.colors['bg_dark'],
                fg=self.colors['fg_light'], font=self.font).grid(row=0, column=0, sticky="w", padx=(0, 5))
        
        self.entry_name = tk.Entry(input_grid, bg=self.colors['bg_light'],
                                   fg=self.colors['fg_light'],
                                   insertbackground=self.colors['fg_light'],
                                   relief=tk.FLAT, bd=3, font=self.font)
        self.entry_name.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        tk.Label(input_grid, text="Телефон:", bg=self.colors['bg_dark'],
                fg=self.colors['fg_light'], font=self.font).grid(row=0, column=2, sticky="w", padx=(15, 5))
        
        self.entry_phone = tk.Entry(input_grid, bg=self.colors['bg_light'],
                                    fg=self.colors['fg_light'],
                                    insertbackground=self.colors['fg_light'],
                                    relief=tk.FLAT, bd=3, font=self.font)
        self.entry_phone.grid(row=0, column=3, sticky="ew", padx=5, pady=5)
        
        input_grid.columnconfigure(1, weight=2)
        input_grid.columnconfigure(3, weight=1)
        
        action_frame = tk.Frame(main_container, bg=self.colors['bg_dark'])
        action_frame.pack(fill=tk.X, pady=(0, 10))
        
        button_frame = tk.Frame(action_frame, bg=self.colors['bg_dark'])
        button_frame.pack(expand=True)
        
        self.create_button(button_frame, "Добавить", self.add_abonent, 'accent').pack(side=tk.LEFT, padx=3)
        self.create_button(button_frame, "Изменить", self.edit_abonent, 'accent').pack(side=tk.LEFT, padx=3)
        self.create_button(button_frame, "Удалить", self.delete_abonent, 'danger').pack(side=tk.LEFT, padx=3)
        self.create_button(button_frame, "Очистить", self.clear_book, 'danger').pack(side=tk.LEFT, padx=3)
        
        search_frame = tk.LabelFrame(main_container, text="Поиск",
                                    bg=self.colors['bg_dark'],
                                    fg=self.colors['fg_light'],
                                    font=self.font_bold,
                                    padx=10, pady=10)
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        search_grid = tk.Frame(search_frame, bg=self.colors['bg_dark'])
        search_grid.pack(fill=tk.X)
        
        tk.Label(search_grid, text="Имя:", bg=self.colors['bg_dark'],
                fg=self.colors['fg_light'], font=self.font).pack(side=tk.LEFT)
        
        self.entry_search = tk.Entry(search_grid, bg=self.colors['bg_light'],
                                     fg=self.colors['fg_light'],
                                     insertbackground=self.colors['fg_light'],
                                     relief=tk.FLAT, bd=3, font=self.font)
        self.entry_search.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.create_button(search_grid, "Найти", self.search_abonent, 'accent').pack(side=tk.LEFT, padx=2)
        self.create_button(search_grid, "Сброс", self.reset_search, 'accent').pack(side=tk.LEFT, padx=2)
        
        table_frame = tk.LabelFrame(main_container, text="Список абонентов",
                                   bg=self.colors['bg_dark'],
                                   fg=self.colors['fg_light'],
                                   font=self.font_bold)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        tree_frame = tk.Frame(table_frame, bg=self.colors['bg_dark'])
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        columns = ("full_name", "phone")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=12)
        
        self.tree.heading("full_name", text="ФИО")
        self.tree.heading("phone", text="Телефон")
        
        self.tree.column("full_name", width=350, anchor="w")
        self.tree.column("phone", width=250, anchor="w")

        scrollbar_y = tk.Scrollbar(tree_frame, orient=tk.VERTICAL, 
                                   command=self.tree.yview,
                                   bg=self.colors['bg_light'],
                                   troughcolor=self.colors['bg_dark'])
        self.tree.configure(yscrollcommand=scrollbar_y.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        file_frame = tk.Frame(main_container, bg=self.colors['bg_dark'])
        file_frame.pack(fill=tk.X)
        
        file_grid = tk.Frame(file_frame, bg=self.colors['bg_dark'])
        file_grid.pack(expand=True)
        
        self.create_button(file_grid, "Сохранить", self.save_book, 'success', 15).pack(side=tk.LEFT, padx=5)
        self.create_button(file_grid, "Загрузить", self.load_book, 'success', 15).pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar()
        self.status_var.set("Готово к работе")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                              bg=self.colors['bg_medium'],
                              fg=self.colors['fg_dim'],
                              bd=0, relief=tk.FLAT, anchor="w",
                              font=self.font_small, padx=10, pady=3)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

    def _refresh_tree(self, selected_index=None):
        for item in self.tree.get_children():
            self.tree.delete(item)

        for idx, a in enumerate(self.abonent_list.to_list()):
            self.tree.insert("", tk.END, iid=str(idx), values=(a.full_name, a.phone))

        if selected_index is not None:
            iid = str(selected_index)
            if iid in self.tree.get_children():
                self.tree.selection_set(iid)
                self.tree.see(iid)

        self.status_var.set(f"Всего записей: {len(self.abonent_list.to_list())}")

    def _get_selected_index(self):
        sel = self.tree.selection()
        if not sel:
            return None
        return int(sel[0])

    def on_tree_select(self, event):
        index = self._get_selected_index()
        if index is None:
            return
        abonent = self.abonent_list.to_list()[index]
        self.entry_name.delete(0, tk.END)
        self.entry_name.insert(0, abonent.full_name)
        self.entry_phone.delete(0, tk.END)
        self.entry_phone.insert(0, abonent.phone)

    def add_abonent(self):
        name = self.entry_name.get().strip()
        phone = self.entry_phone.get().strip()
        if not name or not phone:
            messagebox.showwarning("Внимание", "Заполните и ФИО, и телефон.")
            return
        self.abonent_list.add(Abonent(name, phone))
        self._refresh_tree()
        self.entry_name.delete(0, tk.END)
        self.entry_phone.delete(0, tk.END)
        self.status_var.set(f"Абонент '{name}' добавлен")

    def edit_abonent(self):
        index = self._get_selected_index()
        if index is None:
            messagebox.showinfo("Информация", "Сначала выберите запись для изменения.")
            return
        name = self.entry_name.get().strip()
        phone = self.entry_phone.get().strip()
        if not name or not phone:
            messagebox.showwarning("Внимание", "ФИО и телефон не могут быть пустыми.")
            return
        old_name = self.abonent_list.to_list()[index].full_name
        self.abonent_list.update(index, Abonent(name, phone))
        self._refresh_tree(selected_index=index)
        self.status_var.set(f"Запись '{old_name}' изменена на '{name}'")

    def delete_abonent(self):
        index = self._get_selected_index()
        if index is None:
            messagebox.showinfo("Информация", "Сначала выберите запись для удаления.")
            return
        abonent = self.abonent_list.to_list()[index]
        if messagebox.askyesno("Подтверждение",
                               f"Удалить запись:\n{abonent.full_name} — {abonent.phone}?"):
            self.abonent_list.delete(index)
            self._refresh_tree()
            self.entry_name.delete(0, tk.END)
            self.entry_phone.delete(0, tk.END)
            self.status_var.set(f"Запись '{abonent.full_name}' удалена")

    def clear_book(self):
        if messagebox.askyesno("Подтверждение", "Очистить всю книгу? Это действие нельзя отменить."):
            self.abonent_list.clear()
            self._refresh_tree()
            self.entry_name.delete(0, tk.END)
            self.entry_phone.delete(0, tk.END)
            self.status_var.set("Телефонная книга очищена")

    def search_abonent(self):
        query = self.entry_search.get().strip()
        if not query:
            messagebox.showinfo("Информация", "Введите часть имени для поиска.")
            return
        indices = self.abonent_list.search(query)
        if not indices:
            messagebox.showinfo("Поиск", "Ничего не найдено.")
            return
        self.tree.selection_remove(*self.tree.selection())
        for i in indices:
            iid = str(i)
            if iid in self.tree.get_children():
                self.tree.selection_add(iid)
                self.tree.see(iid)
        self.status_var.set(f"Найдено совпадений: {len(indices)}")

    def reset_search(self):
        self.entry_search.delete(0, tk.END)
        self.tree.selection_remove(*self.tree.selection())
        self.status_var.set(f"Всего записей: {len(self.abonent_list.to_list())}")

    def save_book(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Сохранить книгу"
        )
        if not filename:
            return

        success, error = FileManager.save_phonebook(self.abonent_list, filename)
        if success:
            self.status_var.set(f"Сохранено: {os.path.basename(filename)}")
        else:
            messagebox.showerror("Ошибка сохранения", error)

    def load_book(self):
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Загрузить книгу"
        )
        if not filename:
            return

        success, error = FileManager.load_phonebook(self.abonent_list, filename)
        if success:
            self._refresh_tree()
            self.status_var.set(f"Загружено: {os.path.basename(filename)}")
        else:
            messagebox.showerror("Ошибка загрузки", error)

    def on_close(self):
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PhonebookGUI(root)
    root.mainloop()
