import tkinter as tk
from tkinter import ttk
from control import Control
from datetime import datetime


class ConverterApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Конвертор")
        self.root.geometry("400x450")
        self.root.resizable(False, False)
        
        self.center_window(self.root, 400, 450)
        
        self.bg_color = '#1e1e1e'
        self.fg_color = '#e0e0e0'
        self.entry_bg = '#2d2d2d'
        self.button_bg = '#3a3a3a'
        self.accent_blue = '#0d7377'
        self.accent_green = '#14a76c'
        self.accent_red = '#c23616'
        self.border_color = '#5a5a5a'
        self.border_focus = '#ffffff'
        
        self.root.configure(bg=self.bg_color)
        
        self.ctl = Control()
        self.history = []
        
        self.create_widgets()
    
    def center_window(self, window, width, height):
        window.update_idletasks()
        x = (window.winfo_screenwidth() // 2) - (width // 2)
        y = (window.winfo_screenheight() // 2) - (height // 2)
        window.geometry(f"{width}x{height}+{x}+{y}")
    
    def create_widgets(self):
        menu_bar = tk.Menu(self.root, bg=self.bg_color, fg=self.fg_color,
                          activebackground=self.accent_blue, activeforeground='white',
                          relief='flat', borderwidth=0,
                          font=("sans-serif", 9))
        self.root.config(menu=menu_bar)
        
        menu_bar.add_command(label="Выход", command=self.root.quit)
        menu_bar.add_command(label="История", command=self.show_history)
        menu_bar.add_command(label="Справка", command=self.show_help)
        
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(pady=10, padx=20)
        
        self.entry_input = tk.Entry(main_frame, width=26, font=("monospace", 16),
                                   bg=self.entry_bg, fg=self.fg_color,
                                   relief='flat', bd=0,
                                   insertbackground=self.border_focus,
                                   justify='right',
                                   highlightthickness=1,
                                   highlightbackground=self.border_color,
                                   highlightcolor=self.border_focus)
        self.entry_input.pack(pady=6, ipady=8)
        self.entry_input.bind('<Control-a>', self.select_all_input)
        self.entry_input.bind('<Return>', lambda e: self.convert())
        
        base_frame1 = tk.Frame(main_frame, bg=self.bg_color)
        base_frame1.pack(pady=4)
        
        tk.Label(base_frame1, text="Основание с. сч. исходного числа",
                font=("sans-serif", 9), bg=self.bg_color,
                fg=self.fg_color).pack(side='left', padx=5)
        
        style = ttk.Style()
        style.theme_use('clam')
        
        style.map('Dark.TCombobox',
                  fieldbackground=[('readonly', self.entry_bg)],
                  selectbackground=[('readonly', self.entry_bg)],
                  selectforeground=[('readonly', self.fg_color)])
        
        style.configure('Dark.TCombobox',
                        fieldbackground=self.entry_bg,
                        background=self.button_bg,
                        foreground=self.fg_color,
                        arrowcolor=self.fg_color,
                        bordercolor=self.border_color,
                        lightcolor=self.button_bg,
                        darkcolor=self.button_bg,
                        relief='flat')
        
        self.combo_pin = ttk.Combobox(base_frame1, values=list(range(2, 17)),
                                      width=5, state='readonly',
                                      font=("sans-serif", 9),
                                      style='Dark.TCombobox')
        self.combo_pin.current(8)
        self.combo_pin.pack(side='left', padx=5)
        
        self.root.option_add('*TCombobox*Listbox.background', self.entry_bg)
        self.root.option_add('*TCombobox*Listbox.foreground', self.fg_color)
        self.root.option_add('*TCombobox*Listbox.selectBackground', self.accent_blue)
        self.root.option_add('*TCombobox*Listbox.selectForeground', 'white')
        self.root.option_add('*TCombobox*Listbox.font', ('sans-serif', 9))
        
        self.entry_output = tk.Entry(main_frame, width=26,
                                     font=("monospace", 16, "bold"),
                                     bg=self.entry_bg, fg=self.accent_green,
                                     relief='flat', bd=0,
                                     state='readonly', justify='right',
                                     readonlybackground=self.entry_bg,
                                     highlightthickness=1,
                                     highlightbackground=self.border_color,
                                     highlightcolor=self.border_color)
        self.entry_output.pack(pady=6, ipady=8)
        
        base_frame2 = tk.Frame(main_frame, bg=self.bg_color)
        base_frame2.pack(pady=4)
        
        tk.Label(base_frame2, text="Основание с. сч. результата",
                font=("sans-serif", 9), bg=self.bg_color,
                fg=self.fg_color).pack(side='left', padx=5)
        
        self.combo_pout = ttk.Combobox(base_frame2, values=list(range(2, 17)),
                                       width=5, state='readonly',
                                       font=("sans-serif", 9),
                                       style='Dark.TCombobox')
        self.combo_pout.current(14)
        self.combo_pout.pack(side='left', padx=5)
        
        keypad_frame = tk.Frame(main_frame, bg=self.bg_color)
        keypad_frame.pack(pady=8)
        
        buttons = [
            ['0', '1', '2', '3'],
            ['4', '5', '6', '7'],
            ['8', '9', 'A', 'B'],
            ['C', 'D', 'E', 'F'],
            ['.', 'BS', 'CE', 'Execute']
        ]
        
        for row in buttons:
            row_frame = tk.Frame(keypad_frame, bg=self.bg_color)
            row_frame.pack()
            for btn_text in row:
                if btn_text == 'Execute':
                    bg = self.accent_blue
                    fg = 'white'
                    cmd = self.convert
                elif btn_text in ['BS', 'CE']:
                    bg = self.accent_red
                    fg = 'white'
                    cmd = lambda t=btn_text: self.handle_special(t)
                else:
                    bg = self.button_bg
                    fg = self.fg_color
                    cmd = lambda t=btn_text: self.handle_input(t)
                
                btn = tk.Button(row_frame, text=btn_text, width=6, height=1,
                                bg=bg, fg=fg,
                                font=("sans-serif", 10, "bold"),
                                relief='flat', bd=0, cursor='hand2',
                                activebackground=bg, activeforeground=fg,
                                highlightthickness=1,
                                highlightbackground=self.border_color,
                                highlightcolor=self.border_color,
                                command=cmd)
                btn.pack(side='left', padx=2, pady=2, ipadx=2, ipady=5)
                
                btn.bind('<Enter>', lambda e, b=btn, c=bg: b.configure(bg=self.lighten_color(c)))
                btn.bind('<Leave>', lambda e, b=btn, c=bg: b.configure(bg=c))
    
    def select_all_input(self, event):
        self.entry_input.select_range(0, tk.END)
        return 'break'
    
    def lighten_color(self, color):
        color_map = {
            self.button_bg: '#4a4a4a',
            self.accent_blue: '#0e8589',
            self.accent_red: '#d63a1e'
        }
        return color_map.get(color, color)
    
    def handle_input(self, char):
        if char.isdigit():
            self.ctl.ed.AddDigit(int(char))
        elif char in 'ABCDEF':
            self.ctl.ed.AddDigit(ord(char) - ord('A') + 10)
        elif char == '.':
            self.ctl.ed.AddDelim()
        
        self.entry_input.delete(0, tk.END)
        self.entry_input.insert(0, self.ctl.ed.Number)
    
    def handle_special(self, cmd):
        if cmd == 'BS':
            self.ctl.ed.Bs()
        elif cmd == 'CE':
            self.ctl.ed.Clear()
        
        self.entry_input.delete(0, tk.END)
        self.entry_input.insert(0, self.ctl.ed.Number)
    
    def convert(self):
        try:
            input_str = self.entry_input.get().strip()
            if not input_str:
                self.show_custom_warning("Предупреждение", "Введите число!")
                return
            
            self.ctl.Pin = int(self.combo_pin.get())
            self.ctl.Pout = int(self.combo_pout.get())
            
            self.ctl.ed.number = input_str
            
            result = self.ctl.DoCmnd(19)
            
            if result.startswith("Ошибка"):
                self.show_custom_error("Ошибка", result)
            else:
                self.entry_output.config(state='normal')
                self.entry_output.delete(0, tk.END)
                self.entry_output.insert(0, result)
                self.entry_output.config(state='readonly')
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.history.append(
                    f"[{timestamp}] {input_str} (p={self.ctl.Pin}) → {result} (p={self.ctl.Pout})"
                )
        
        except Exception as e:
            self.show_custom_error("Ошибка", str(e))
    
    def show_history(self):
        history_window = tk.Toplevel(self.root)
        history_window.title("История преобразований")
        history_window.resizable(False, False)
        history_window.configure(bg=self.bg_color)
        
        self.center_window(history_window, 500, 400)
        
        tk.Label(history_window, text="История преобразований",
                 font=("sans-serif", 12, "bold"),
                 bg=self.bg_color, fg=self.fg_color).pack(pady=10)
        
        text_frame = tk.Frame(history_window, bg=self.bg_color)
        text_frame.pack(padx=15, pady=10, fill='both', expand=True)
        
        scrollbar = tk.Scrollbar(text_frame, bg=self.button_bg,
                                 troughcolor=self.bg_color,
                                 activebackground=self.border_color)
        scrollbar.pack(side='right', fill='y')
        
        text_area = tk.Text(text_frame, width=60, height=15,
                            bg=self.entry_bg, fg=self.fg_color,
                            font=("monospace", 9),
                            relief='flat', bd=0,
                            wrap=tk.WORD,
                            yscrollcommand=scrollbar.set)
        text_area.pack(side='left', fill='both', expand=True)
        
        scrollbar.config(command=text_area.yview)
        
        if self.history:
            for entry in self.history:
                text_area.insert(tk.END, entry + "\n")
        else:
            text_area.insert(tk.END, "История пуста")
        
        text_area.config(state='disabled')
        
        btn_frame = tk.Frame(history_window, bg=self.bg_color)
        btn_frame.pack(pady=15)
        
        clear_btn = tk.Button(btn_frame, text="Очистить историю",
                              command=lambda: self.clear_history(text_area),
                              bg=self.accent_red, fg='white',
                              font=("sans-serif", 10, "bold"),
                              relief='flat', bd=0, cursor='hand2',
                              width=18, height=1)
        clear_btn.pack(side='left', padx=5, ipady=5)
        
        close_btn = tk.Button(btn_frame, text="Закрыть",
                              command=history_window.destroy,
                              bg=self.button_bg, fg=self.fg_color,
                              font=("sans-serif", 10, "bold"),
                              relief='flat', bd=0, cursor='hand2',
                              width=12, height=1)
        close_btn.pack(side='left', padx=5, ipady=5)
    
    def clear_history(self, text_area):
        self.history.clear()
        text_area.config(state='normal')
        text_area.delete(1.0, tk.END)
        text_area.insert(tk.END, "История пуста")
        text_area.config(state='disabled')
    
    def show_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.title("Справка")
        help_window.configure(bg=self.bg_color)
        help_window.resizable(False, False)
        
        self.center_window(help_window, 400, 240)
        
        tk.Label(help_window, text="Конвертор систем счисления",
                 font=("sans-serif", 14, "bold"),
                 bg=self.bg_color, fg=self.fg_color).pack(pady=20)
        
        info_text = (
            "Поддерживает преобразование между\n"
            "системами счисления от 2 до 16\n\n"
        )
        
        tk.Label(help_window, text=info_text,
                 font=("sans-serif", 10),
                 bg=self.bg_color, fg=self.fg_color,
                 justify='center').pack(pady=15)
        
        ok_btn = tk.Button(help_window, text="OK",
                           command=help_window.destroy,
                           bg=self.accent_blue, fg='white',
                           font=("sans-serif", 10, "bold"),
                           relief='flat', bd=0, cursor='hand2',
                           padx=40, pady=8,
                           )
        ok_btn.pack(pady=15)
        
        help_window.bind('<Return>', lambda e: help_window.destroy())
        help_window.bind('<Escape>', lambda e: help_window.destroy())
     
    def show_custom_warning(self, title, message):
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.configure(bg=self.bg_color)
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        self.center_window(dialog, 350, 200)
        
        tk.Label(dialog, text="⚠", font=("sans-serif", 30),
                 bg=self.bg_color, fg='#f39c12').pack(pady=10)
        
        if len(message) > 50:
            display_message = message[:47] + "..."
        else:
            display_message = message
        
        tk.Label(dialog, text=display_message, font=("sans-serif", 10),
                 bg=self.bg_color, fg=self.fg_color, 
                 wraplength=300).pack(pady=5)
        
        ok_btn = tk.Button(dialog, text="OK", command=dialog.destroy,
                           bg=self.accent_blue, fg='white',
                           font=("sans-serif", 10, "bold"),
                           relief='flat', bd=0, cursor='hand2',
                           padx=40, pady=8)
        ok_btn.pack(pady=15)
        
        dialog.bind('<Return>', lambda e: dialog.destroy())
        dialog.bind('<Escape>', lambda e: dialog.destroy())

    def show_custom_error(self, title, message):
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.configure(bg=self.bg_color)
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        self.center_window(dialog, 350, 200)
        
        tk.Label(dialog, text="✖", font=("sans-serif", 30),
                 bg=self.bg_color, fg=self.accent_red).pack(pady=10)
        
        if len(message) > 50:
            display_message = message[:47] + "..."
        else:
            display_message = message
        
        tk.Label(dialog, text=display_message, font=("sans-serif", 10),
                 bg=self.bg_color, fg=self.fg_color,
                 wraplength=300).pack(pady=5)
        
        ok_btn = tk.Button(dialog, text="OK", command=dialog.destroy,
                           bg=self.accent_blue, fg='white',
                           font=("sans-serif", 10, "bold"),
                           relief='flat', bd=0, cursor='hand2',
                           padx=40, pady=8)
        ok_btn.pack(pady=15)
        
        dialog.bind('<Return>', lambda e: dialog.destroy())
        dialog.bind('<Escape>', lambda e: dialog.destroy())


if __name__ == "__main__":
    root = tk.Tk()
    app = ConverterApp(root)
    root.mainloop()

