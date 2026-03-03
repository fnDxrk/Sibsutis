from anumber import TPNumber, TFrac, TComp
from aeditor import AEditor, PEditor, FEditor, CEditor
from memory import TMemory
from processor import TProc, TOprtn, TFunc
import json
import os

HISTORY_FILE = "history.json"

class TCtrl:

    def __init__(self, mode: str = "p"):
        self.editor = FEditor()
        zero_num = TFrac("0")
        self.processor = TProc(zero_num, zero_num)
        self.memory = TMemory(zero_num)
        self._pending_op = None
        self._pending_num = None
        self._last_operand = None
        self._need_clear_editor = False
        self._repeated = False
        self.history = self._load_history()

    @property
    def display(self):
        return self.editor.string

    def do_editor_command(self, cmd: int):
        if self._need_clear_editor:
            self.editor.clear()
            self._need_clear_editor = False
        self.editor.edit(cmd)
        return self.editor.string

    def _get_current_number(self):
        editor_str = self.editor.string.strip() or "0"
        return TFrac(editor_str)

    def do_calc_command(self, cmd: str):
        try:
            if cmd in ('sqr', 'inv'):
                num = self._get_current_number()
                self.processor.rop = num
                
                if cmd == 'sqr':
                    self.processor.func_run(TFunc.SQR)
                else:
                    self.processor.func_run(TFunc.REV)
                
                if self.processor.error:
                    self.editor.string = f"ERR: {self.processor.error}"
                    return self.editor.string
                
                result = self.processor.rop
                self.editor.string = result.string
                self._need_clear_editor = False
                
                return self.editor.string

            if cmd in '+-*/':
                current_num = self._get_current_number()

                if self._pending_op is not None and self._pending_num is not None:
                    self.processor.lop_res = self._pending_num
                    self.processor.rop = current_num
                    self.processor.operation = self._str_to_oprtn(self._pending_op)
                    self.processor.oprtn_run()
                    
                    if self.processor.error:
                        self.editor.string = f"ERR: {self.processor.error}"
                        self._reset_state()
                        return self.editor.string
                    
                    result = self.processor.lop_res
                    self._pending_num = result
                    self.editor.string = result.string
                else:
                    self._pending_num = current_num

                self._pending_op = cmd
                self._last_operand = current_num
                self._need_clear_editor = True
                
                return self.editor.string

            if cmd == '=':
                current_num = self._get_current_number()

                if self._pending_op is not None and self._pending_num is not None:
                    if not self._repeated:
                        if not self._need_clear_editor:
                            self._last_operand = current_num
                        elif self._last_operand is None:
                            self._last_operand = current_num
                    else:
                        current_num = self._last_operand

                    self.processor.lop_res = self._pending_num
                    self.processor.rop = current_num
                    self.processor.operation = self._str_to_oprtn(self._pending_op)
                    self.processor.oprtn_run()

                    result = self.processor.lop_res
                    self.history.append(f"{self._pending_num.string} {self._pending_op} {self._last_operand.string} = {result.string}")
                    self.editor.string = result.string
                    self._pending_num = result

                    if self._repeated:
                        pass
                    else:
                        self._repeated = True

                    self._need_clear_editor = False
                    return self.editor.string

            if cmd == 'C':
                self.editor.clear()
                self._reset_state()
                return "0"

            return self.editor.string

        except Exception as e:
            self.editor.string = f"ERR: {e}"
            self._reset_state()
            return self.editor.string

    def _reset_state(self):
        self._pending_op = None
        self._pending_num = None
        self._last_operand = None
        self._repeated = False
        self._need_clear_editor = False
        self.processor.oprtn_clear()
        self.processor.clear_error()

    def _str_to_oprtn(self, op: str) -> TOprtn:
        return {
            '+': TOprtn.ADD,
            '-': TOprtn.SUB,
            '*': TOprtn.MUL,
            '/': TOprtn.DVD
        }[op]

    def do_memory_command(self, cmd: str):
        try:
            num = self._get_current_number()

            if cmd == 'MS':
                self.memory.mem_store(num)
            elif cmd == 'M+':
                self.memory.mem_add(num)
            elif cmd == 'MR':
                restored = self.memory.mem_restore()
                if self._need_clear_editor:
                    self.editor.clear()
                self.editor.string = restored.string
                self._need_clear_editor = False
            elif cmd == 'MC':
                self.memory.mem_clear()

            return self.editor.string
        except Exception as e:
            return f"ERR: {e}"

    def set_base(self, base: int):
        self.base = base
        self.editor.clear()
        zero_num = TPNumber("0", base)
        self.processor = TProc(zero_num, zero_num)
        self.memory = TMemory(zero_num)
        self._reset_state()

    def _load_history(self):
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def save_history(self):
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False)
